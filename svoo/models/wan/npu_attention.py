# Portions of WanNPUProcessor are adapted from Hugging Face Diffusers.
# Copyright 2025 The Wan Team and The HuggingFace Team.
# Licensed under Apache-2.0; see the project LICENSE.
"""Portable SVOO reference backend; no CUDA, Triton or FlashInfer imports.

The gathered-cluster implementation prioritizes correctness over throughput.
CPU execution is supported for tests; production inference uses torch_npu.
"""
import csv

import torch
import torch.nn.functional as F
from svoo.routing import identify_dynamic_map


def dense_attention(q, k, v):
    """Inputs and output use BNSD. No masks, dropout, or causal attention."""
    if q.device.type == "npu":
        import torch_npu
        return torch_npu.npu_fusion_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            head_num=q.shape[1], input_layout="BNSD",
            scale=q.shape[-1] ** -0.5, keep_prob=1.0,
            pre_tockens=2147483647, next_tockens=2147483647,
            sparse_mode=0,
        )[0]
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)


def _means(x, labels, old):
    b, n, d = x.shape
    sums = torch.zeros(b, old.shape[1], d, device=x.device, dtype=torch.float32)
    counts = torch.zeros(b, old.shape[1], device=x.device, dtype=torch.int32)
    sorted_labels, order = labels.sort(dim=-1)
    sorted_x = x.gather(1, order[..., None].expand(-1, -1, d))
    sums.scatter_add_(1, sorted_labels[..., None].expand(-1, -1, d), sorted_x.float())
    counts.scatter_add_(1, sorted_labels, torch.ones(b, n, device=x.device, dtype=torch.int32))
    means = sums / counts.float().clamp_min(1)[..., None]
    return torch.where(counts[..., None] > 0, means, old.float()).to(x.dtype), counts


def _profile_assign(x, basis, centers, chunk):
    # Match the original normalized dot-product profile assignment while
    # bounding the temporary tensor to [batch_heads, chunk, clusters].
    # Original centroid-centroid matmul rounds to the token dtype BEFORE
    # FP32 normalization. Token profiles below are computed in FP32.
    centers_profile = (centers @ basis.transpose(-1, -2)).float()
    centers_profile = centers_profile / (
        centers_profile.square().sum(-1, keepdim=True) + 1e-8
    ).sqrt()
    labels = []
    for start in range(0, x.shape[1], chunk):
        profile = x[:, start:start + chunk].float() @ basis.float().transpose(-1, -2)
        profile = profile / (profile.square().sum(-1, keepdim=True) + 1e-8).sqrt()
        distance = 2.0 - 2.0 * (profile @ centers_profile.transpose(-1, -2))
        labels.append(distance.argmin(-1))
    return torch.cat(labels, dim=1)


def co_cluster(q, k, qc, kc, iters=2, chunk=256, q_indices=None, k_indices=None):
    if min(qc, kc, iters, chunk) < 1:
        raise ValueError("Cluster counts, iterations and chunk size must be positive")
    b, n, d = q.shape
    if q.shape != k.shape:
        raise ValueError("Original SVOO co-clustering requires matching Q/K shapes")
    # Sampling with replacement is intentional; do not cap clusters by N.
    qi = torch.randint(n, (b, qc), device=q.device) if q_indices is None else q_indices.to(q.device)
    ki = torch.randint(n, (b, kc), device=k.device) if k_indices is None else k_indices.to(k.device)
    qcent = q.gather(1, qi[..., None].expand(-1, -1, d))
    kcent = k.gather(1, ki[..., None].expand(-1, -1, d))
    for _ in range(iters):
        kl = _profile_assign(k, qcent, kcent, chunk)
        kcent, ks = _means(k, kl, kcent)
        ql = _profile_assign(q, kcent, qcent, chunk)
        qcent, qs = _means(q, ql, qcent)
    return ql, qcent, qs, kl, kcent, ks


def select_blocks(qcent, kcent, ks, top_p, min_ratio):
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    # Flattened batch-head dimension is treated as H. Wan SVOO has B=1.
    # Invoke the ORIGINAL policy, including BF16 softmax rounding, shifted
    # cumulative top-p boundary and integer minimum-cluster truncation.
    return identify_dynamic_map(
        qcent.unsqueeze(0), kcent.unsqueeze(0), None, ks.unsqueeze(0),
        top_p, min_ratio,
    )[0]


def gathered_sparse_attention(q, k, v, ql, kl, blocks):
    """Exactly evaluate the selected token blocks, preserving original order."""
    b, h, n, d = q.shape
    qf, kf, vf = (x.reshape(b * h, x.shape[2], d) for x in (q, k, v))
    out = torch.empty_like(qf)
    # Moving only small routing metadata to CPU avoids per-cluster NPU nonzero
    # synchronization. Large token tensors remain on the original device.
    qcpu, kcpu, mcpu = ql.cpu(), kl.cpu(), blocks.cpu()
    # Match the original label argsort before gathering Q/K/V. Within-cluster
    # tie order remains backend dependent, as in torch.argsort itself.
    qorder, korder = ql.argsort(dim=-1).cpu(), kl.argsort(dim=-1).cpu()
    for bh in range(b * h):
        for cluster in range(blocks.shape[1]):
            qi = qorder[bh][qcpu[bh, qorder[bh]] == cluster]
            if qi.numel() == 0:
                continue
            ki = korder[bh][mcpu[bh, cluster][kcpu[bh, korder[bh]]]]
            if ki.numel() == 0:
                raise RuntimeError("Routing selected no nonempty key cluster")
            qi, ki = qi.to(q.device), ki.to(q.device)
            output = dense_attention(
                qf[bh].index_select(0, qi)[None, None],
                kf[bh].index_select(0, ki)[None, None],
                vf[bh].index_select(0, ki)[None, None],
            )[0, 0]
            out[bh].index_copy_(0, qi, output)
    return out.reshape(b, h, n, d)


def load_profile(path):
    if path is None:
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {
            (int(r["Step"]), int(r["Layer"]), int(r["Head"])): float(r["Sparsity"])
            for r in csv.DictReader(f)
        }


class WanNPUProcessor:
    # QKV projection, normalization, RoPE and output follow diffusers 0.36.0
    # WanAttnProcessor (Apache-2.0), with a separate attention backend.
    _attention_backend = None
    _parallel_config = None

    def __init__(self, layer, sparse=False, qc=256, kc=1024, iters=2,
                 chunk=256, top_p=0.9, min_ratio=0.1, profile=None,
                 ratio_min=0.15, ratio_max=0.20, fp_steps=10, fp_layers=1,
                 reuse_start=11, reuse_interval=20, iters_init=None, iters_step=None,
                 fp_timestep=None):
        self.layer, self.sparse = layer, sparse
        self.qc, self.kc, self.iters, self.chunk = qc, kc, iters, chunk
        self.top_p, self.min_ratio = top_p, min_ratio
        self.profile = profile or {}
        self.ratio_min, self.ratio_max = ratio_min, ratio_max
        self.fp_steps, self.fp_layers = fp_steps, fp_layers
        self.reuse_start, self.reuse_interval = reuse_start, reuse_interval
        self.step, self.cache = 1, None
        self.iters_init = iters if iters_init is None else iters_init
        self.iters_step = iters if iters_step is None else iters_step
        self.initialized = False
        self.fp_timestep, self.current_timestep = fp_timestep, None

    def _ratios(self, b, h):
        keys = [(self.step, self.layer, head) for head in range(h)]
        if not all(key in self.profile for key in keys):
            return self.min_ratio
        return [
            min(self.ratio_max, max(self.ratio_min, self.profile[key]))
            for _ in range(b) for key in keys
        ]

    def _self_attention(self, q, k, v):
        if self.sparse and q.shape[0] != 1:
            raise ValueError("Original Wan SVOO requires batch size 1")
        if self.fp_timestep is not None and self.current_timestep is None:
            raise RuntimeError("Scheduler timestep is required for original SVOO warmup")
        warmup = (self.current_timestep > self.fp_timestep if self.fp_timestep is not None
                  else self.step <= self.fp_steps)
        if not self.sparse or warmup or self.layer < self.fp_layers:
            return dense_attention(q, k, v)
        b, h, n, d = q.shape
        recluster = (
            self.cache is None or self.cache[0] != tuple(q.shape)
            or self.step < self.reuse_start
            or (self.step - self.reuse_start) % self.reuse_interval == 0
        )
        if recluster:
            result = co_cluster(q.reshape(b*h, n, d), k.reshape(b*h, n, d),
                                self.qc, self.kc,
                                self.iters_step if self.initialized else self.iters_init, self.chunk)
            self.initialized = True
            self.cache = (tuple(q.shape), result)
        ql, qcent, _, kl, kcent, ks = self.cache[1]
        blocks = select_blocks(qcent, kcent, ks, self.top_p, self._ratios(b, h))
        return gathered_sparse_attention(q, k, v, ql, kl, blocks)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        if attention_mask is not None:
            raise NotImplementedError("NPU Wan processor currently supports unmasked attention only")
        image = None
        if attn.add_k_proj is not None:
            image_len = encoder_hidden_states.shape[1] - 512
            image, encoder_hidden_states = (
                encoder_hidden_states[:, :image_len],
                encoder_hidden_states[:, image_len:],
            )
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if getattr(attn, "fused_projections", False):
            if attn.cross_attention_dim_head is None:
                q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
                q = attn.to_q(hidden_states)
                k, v = attn.to_kv(context).chunk(2, dim=-1)
        else:
            q, k, v = attn.to_q(hidden_states), attn.to_k(context), attn.to_v(context)
        q, k = attn.norm_q(q), attn.norm_k(k)
        q, k, v = (x.unflatten(2, (attn.heads, -1)) for x in (q, k, v))
        if rotary_emb is not None:
            cos, sin = rotary_emb[0][..., 0::2], rotary_emb[1][..., 1::2]
            def rotate(x):
                a, b = x.unflatten(-1, (-1, 2)).unbind(-1)
                return torch.stack((a*cos - b*sin, a*sin + b*cos), -1).flatten(-2).to(x.dtype)
            q, k = rotate(q), rotate(k)
        q, k, v = (x.transpose(1, 2).contiguous() for x in (q, k, v))
        out = self._self_attention(q, k, v) if encoder_hidden_states is None else dense_attention(q, k, v)
        if image is not None:
            if getattr(attn, "fused_projections", False):
                ik, iv = attn.to_added_kv(image).chunk(2, dim=-1)
            else:
                ik, iv = attn.add_k_proj(image), attn.add_v_proj(image)
            ik = attn.norm_added_k(ik)
            ik, iv = (x.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
                      for x in (ik, iv))
            out = out + dense_attention(q, ik, iv)
        out = out.transpose(1, 2).flatten(2).to(q.dtype)
        return attn.to_out[1](attn.to_out[0](out))


def install_processors(pipe, mode, **kwargs):
    processors = []
    for transformer in (pipe.transformer, getattr(pipe, "transformer_2", None)):
        if transformer is None:
            continue
        for layer, block in enumerate(transformer.blocks):
            for name in ("attn1", "attn2"):
                processor = WanNPUProcessor(layer, sparse=mode == "svoo" and name == "attn1", **kwargs)
                getattr(block, name).set_processor(processor)
                processors.append(processor)
    return processors
