"""Run: python -m unittest discover -s tests -p test_npu_attention.py -v
On Ascend: SVOO_TEST_DEVICE=npu:0 python -m unittest discover -s tests -p test_npu_attention.py -v
"""
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

DEVICE = os.environ.get("SVOO_TEST_DEVICE", "cpu")
if DEVICE.startswith("npu"):
    import torch_npu
    torch.npu.set_device(DEVICE)

from svoo.models.wan.npu_attention import (
    WanNPUProcessor, _profile_assign, co_cluster, dense_attention,
    gathered_sparse_attention, select_blocks,
)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.dtype = torch.bfloat16 if (DEVICE.startswith("npu") or os.environ.get("SVOO_TEST_DTYPE") == "bf16") else torch.float32
        self.q, self.k, self.v = [
            torch.randn(2, 2, 33, 128).to(DEVICE, self.dtype) for _ in range(3)
        ]

    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual.cpu().float(), expected.cpu().float(),
                                   atol=0.025 if self.dtype == torch.bfloat16 else 2e-5,
                                   rtol=0.025 if self.dtype == torch.bfloat16 else 2e-5)

    def test_dense_matches_cpu_sdpa(self):
        expected = F.scaled_dot_product_attention(
            self.q.cpu().float(), self.k.cpu().float(), self.v.cpu().float())
        self.assert_close(dense_attention(self.q, self.k, self.v), expected)

    def test_sparse_matches_explicit_token_mask(self):
        ql, kl = [torch.randint(0, 4, (4, 33)).to(DEVICE) for _ in range(2)]
        blocks = torch.rand(4, 4, 4) > 0.5
        # Every query has at least one nonempty selected key cluster.
        blocks[:, :, 0] = True
        kl[:, 0] = 0
        result = gathered_sparse_attention(self.q, self.k, self.v, ql, kl, blocks.to(DEVICE))
        mask = torch.stack([
            blocks[i][ql[i].cpu()][:, kl[i].cpu()] for i in range(4)
        ]).reshape(2, 2, 33, 33)
        expected = F.scaled_dot_product_attention(
            self.q.cpu().float(), self.k.cpu().float(), self.v.cpu().float(), attn_mask=mask)
        self.assert_close(result, expected)

    def test_all_blocks_match_dense(self):
        ql, _, _, kl, _, _ = co_cluster(
            self.q.reshape(4, 33, 128), self.k.reshape(4, 33, 128), 4, 7, 2, 9)
        blocks = torch.ones(4, 4, 7, dtype=torch.bool, device=DEVICE)
        self.assert_close(
            gathered_sparse_attention(self.q, self.k, self.v, ql, kl, blocks),
            dense_attention(self.q, self.k, self.v),
        )

    def test_chunked_clustering_and_empty_clusters(self):
        # Same initialization, different chunking must produce identical routing.
        torch.manual_seed(23)
        a = co_cluster(self.q.reshape(4, 33, 128), self.k.reshape(4, 33, 128), 4, 7, 2, 8)
        torch.manual_seed(23)
        b = co_cluster(self.q.reshape(4, 33, 128), self.k.reshape(4, 33, 128), 4, 7, 2, 33)
        torch.testing.assert_close(a[0], b[0])
        torch.testing.assert_close(a[3], b[3])
        torch.testing.assert_close(a[2].sum(1), torch.full((4,), 33, device=DEVICE, dtype=torch.int64))
        torch.testing.assert_close(a[5].sum(1), torch.full((4,), 33, device=DEVICE, dtype=torch.int64))
        zero = torch.zeros(1, 9, 256, device=DEVICE, dtype=self.dtype)
        result = co_cluster(zero, zero, 4, 7, 2, 3)
        blocks = select_blocks(result[1], result[4], result[5], 0.9, 0.1)
        self.assertTrue(bool(blocks[..., 0].all()))
        self.assertTrue(bool(torch.isfinite(result[1]).all()))

    def test_block_selection_weighted_mass_and_minimum(self):
        qcent = torch.zeros(2, 3, 128, device=DEVICE)
        kcent = torch.zeros(2, 4, 128, device=DEVICE)
        ks = torch.tensor([[8., 1., 1., 0.], [1., 1., 1., 1.]], device=DEVICE)
        blocks = select_blocks(qcent, kcent, ks, 0.7, [0.5, 0.75])
        torch.testing.assert_close(blocks.sum(-1).cpu(), torch.tensor([[2, 2, 2], [3, 3, 3]]))
        self.assertFalse(bool(blocks[0, :, 3].any()))
        self.assertTrue(bool(select_blocks(qcent, kcent, ks, 1, 0).all()))

    def test_warmup_and_reuse(self):
        q, k, v = self.q[:1], self.k[:1], self.v[:1]
        p = WanNPUProcessor(1, sparse=True, qc=4, kc=7, chunk=8,
                            fp_steps=1, fp_layers=1, reuse_start=3, reuse_interval=2)
        self.assert_close(p._self_attention(q, k, v),
                          dense_attention(q, k, v))
        self.assertIsNone(p.cache)
        p.step = 2
        p._self_attention(q, k, v)
        p.step = 3
        p._self_attention(q, k, v)
        cache = p.cache
        p.step = 4
        with patch("svoo.models.wan.npu_attention.co_cluster", side_effect=AssertionError("Unexpected recluster")):
            p._self_attention(q, k, v)
        self.assertIs(p.cache, cache)

    def test_wan_processor_dense_matches_diffusers(self):
        from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor
        attn = WanAttention(dim=256, heads=2, dim_head=128).eval().to(DEVICE, self.dtype)
        hidden = torch.randn(1, 17, 256).to(DEVICE, self.dtype)
        context = torch.randn(1, 9, 256).to(DEVICE, self.dtype)
        # CPU official processor provides an independent projection/RoPE/output reference.
        import copy
        reference_attn = copy.deepcopy(attn).cpu().float()
        reference_attn.set_processor(WanAttnProcessor())
        processor = WanNPUProcessor(0)
        angles = torch.randn(1, 17, 1, 64).to(DEVICE)
        cos = angles.cos().repeat_interleave(2, dim=-1)
        sin = angles.sin().repeat_interleave(2, dim=-1)
        rotary = (cos, sin)
        for ctx, rope in ((None, None), (None, rotary), (context, None)):
            actual = processor(attn, hidden, ctx, rotary_emb=rope)
            expected = reference_attn(
                hidden.cpu().float(),
                encoder_hidden_states=None if ctx is None else ctx.cpu().float(),
                rotary_emb=None if rope is None else tuple(x.cpu() for x in rope))
            self.assert_close(actual, expected)


    def test_tiny_wan_transformer_forward(self):
        import copy
        from types import SimpleNamespace
        from diffusers import WanTransformer3DModel
        from svoo.models.wan.npu_attention import install_processors
        reference = WanTransformer3DModel(
            patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=128,
            in_channels=4, out_channels=4, text_dim=32, freq_dim=32,
            ffn_dim=64, num_layers=2,
        ).eval()
        model = copy.deepcopy(reference).to(DEVICE, self.dtype)
        pipe = SimpleNamespace(transformer=model)
        hidden = torch.randn(1, 4, 3, 4, 4).to(DEVICE, self.dtype)
        context = torch.randn(1, 9, 32).to(DEVICE, self.dtype)
        timestep = torch.tensor([500.], device=DEVICE)
        processors = install_processors(pipe, "dense")
        with torch.inference_mode():
            expected = reference(hidden.cpu().float(), timestep.cpu(), context.cpu().float()).sample
            actual = model(hidden, timestep, context).sample
        self.assert_close(actual, expected)
        processors = install_processors(
            pipe, "svoo", qc=4, kc=8, iters=2, chunk=4, fp_steps=0, fp_layers=0,
        )
        with torch.inference_mode():
            sparse = model(hidden, timestep, context).sample
        self.assertEqual(sparse.shape, actual.shape)
        self.assertTrue(bool(torch.isfinite(sparse).all()))
        self.assertTrue(all(p.cache is not None for p in processors if p.sparse))

    def test_original_cluster_count_and_dtype(self):
        q = self.q[:1, :1, :5].reshape(1, 5, 128)
        k = self.k[:1, :1, :5].reshape(1, 5, 128)
        result = co_cluster(q, k, 9, 11, 2, 3)
        self.assertEqual(result[1].shape, (1, 9, 128))
        self.assertEqual(result[4].shape, (1, 11, 128))
        self.assertEqual(result[1].dtype, self.dtype)
        self.assertEqual(result[4].dtype, self.dtype)
        self.assertEqual(result[2].dtype, torch.int32)
        self.assertEqual(result[5].dtype, torch.int32)

    def test_original_bf16_top_p_rounding(self):
        # Original BF16 weighted softmax/cumsum rounds the second cumulative
        # probability and threshold to the same BF16 value. A FP32 rewrite
        # would drop the third cluster and therefore change the routing.
        centers = torch.zeros(1, 3, 128, dtype=torch.bfloat16, device=DEVICE)
        mask = select_blocks(centers[:, :1], centers,
                             torch.ones(1, 3, device=DEVICE, dtype=torch.int32),
                             0.6661, 0)
        self.assertTrue(bool(mask.all()))

    def test_bf16_top_p_boundary_matches_cpu(self):
        # Compare with the original CPU scalar policy across both sides of
        # the BF16 rounding boundary and several batch/head shapes.
        for heads in (1, 2, 12):
            centers = torch.zeros(heads, 3, 128, dtype=torch.bfloat16)
            counts = torch.ones(heads, 3, dtype=torch.int32)
            for threshold in (0.664, 0.6661, 0.668, 0.67):
                with self.subTest(heads=heads, threshold=threshold):
                    expected = select_blocks(centers[:, :1], centers, counts, threshold, 0)
                    actual = select_blocks(centers[:, :1].to(DEVICE), centers.to(DEVICE),
                                           counts.to(DEVICE), threshold, 0)
                    torch.testing.assert_close(actual.cpu(), expected)

    def test_top_p_one_still_runs_original_clustering(self):
        q, k, v = self.q[:1], self.k[:1], self.v[:1]
        p = WanNPUProcessor(0, sparse=True, qc=4, kc=7,
                            fp_steps=0, fp_layers=0, top_p=1)
        output = p._self_attention(q, k, v)
        self.assertIsNotNone(p.cache)
        self.assert_close(output, dense_attention(q, k, v))

    def test_scheduler_timestep_controls_warmup(self):
        q, k, v = self.q[:1], self.k[:1], self.v[:1]
        p = WanNPUProcessor(0, sparse=True, qc=4, kc=7, fp_steps=100,
                            fp_layers=0, fp_timestep=500)
        p.current_timestep = 501
        p._self_attention(q, k, v)
        self.assertIsNone(p.cache)
        p.current_timestep = 500
        p._self_attention(q, k, v)
        self.assertIsNotNone(p.cache)

    def test_original_entrypoint_defaults(self):
        import wan_npu_inference
        with patch("sys.argv", ["wan_npu_inference.py", "--model_id", "local"]):
            args = wan_npu_inference.parse_args()
        self.assertEqual((args.height, args.width, args.num_frames,
                          args.num_inference_steps), (720, 1280, 81, 50))
        self.assertEqual((args.kmeans_iter_init, args.kmeans_iter_step), (2, 2))
        self.assertIsNotNone(args.sparsity_csv_path)

if __name__ == "__main__":
    unittest.main()
