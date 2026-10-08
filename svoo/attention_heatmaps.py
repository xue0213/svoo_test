"""Mean-pooled pre-softmax QK score views; never materialize N x N scores."""
import json
import random
from pathlib import Path

import numpy as np
import torch


def pooled_scores(q, k, block_size, q_order=None, k_order=None):
    """FP32 score block means via bilinearity, including partial final blocks.

    Input vectors have already undergone QK normalization and RoPE. Pooling
    uses FP32 analysis arithmetic, not a reproduction of BF16 kernel rounding.
    """
    if block_size < 1 or q.ndim != 2 or k.ndim != 2 or q.shape[-1] != k.shape[-1]:
        raise ValueError("Expected [tokens, dim] Q/K and a positive block size")

    def pool(x, order):
        x = x.detach().to(device="cpu", dtype=torch.float32)
        if order is not None:
            x = x.index_select(0, order.detach().cpu().long())
        full = x.shape[0] // block_size
        pieces = []
        if full:
            pieces.append(x[:full * block_size].reshape(full, block_size, -1).mean(1))
        if full * block_size < x.shape[0]:
            pieces.append(x[full * block_size:].mean(0, keepdim=True))
        if not pieces:
            raise ValueError("Cannot pool an empty token sequence")
        return torch.cat(pieces)

    return (pool(q, q_order) @ pool(k, k_order).T) * (q.shape[-1] ** -0.5)


def cluster_boundaries(labels, order):
    sorted_labels = labels.cpu().long()[order.cpu().long()]
    changes = torch.nonzero(sorted_labels[1:] != sorted_labels[:-1]).flatten() + 1
    return torch.cat((torch.tensor([0]), changes, torch.tensor([labels.numel()])))


class AttentionHeatmaps:
    def __init__(self, output_dir, num_layers, seed=0, layer_count=7,
                 branch="positive", step=None, metadata=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if num_layers < 1 or layer_count < 1 or branch not in ("positive", "negative", "both"):
            raise ValueError("Invalid heatmap layer count or CFG branch")
        self.layers = sorted(random.Random(seed).sample(range(num_layers), min(layer_count, num_layers)))
        self.branches = {"positive", "negative"} if branch == "both" else {branch}
        self.step = step
        self.done = set()
        self.metadata = dict(metadata or {}, selected_layers=self.layers, layer_seed=seed,
                             branches=sorted(self.branches), requested_step=step,
                             score="FP32 mean of post-RoPE QK^T / sqrt(head_dim), before softmax",
                             pooling="original latent-frame blocks; sorted equal-token-count blocks")
        self.records = []
        # Validate plotting dependencies when capture is enabled.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self._manifest()
        print(f"[Heatmaps] layers (zero-based)={self.layers}, branches={sorted(self.branches)}", flush=True)

    def _manifest(self):
        (self.output_dir / "manifest.json").write_text(
            json.dumps(dict(self.metadata, captured=self.records), indent=2), encoding="utf-8")

    def capture(self, layer, step, branch, q, k, q_labels, k_labels, grid):
        key = (layer, branch)
        if layer not in self.layers or branch not in self.branches or key in self.done:
            return
        if self.step is not None and step != self.step:
            return
        if q.shape[0] != 1 or q.shape != k.shape:
            raise ValueError("Heatmaps require matching Q/K and batch size one")
        frames, height, width = grid
        if frames * height * width != q.shape[2]:
            raise ValueError(f"Token grid {grid} does not match Q shape {tuple(q.shape)}")
        block_size = height * width
        # Use the same backend argsort policy as gathered_sparse_attention.
        qo, ko = q_labels.argsort(-1).cpu(), k_labels.argsort(-1).cpu()
        target = self.output_dir / f"step_{step:03d}" / branch / f"layer_{layer:02d}"
        target.mkdir(parents=True, exist_ok=True)
        print(f"[Heatmaps] step={step} layer={layer} branch={branch} grid={grid} begin", flush=True)
        for head in range(q.shape[1]):
            # Transfer one head at a time. No Q/K values or routing are modified.
            qh, kh = q[0, head].detach().cpu(), k[0, head].detach().cpu()
            before = pooled_scores(qh, kh, block_size).numpy()
            after = pooled_scores(qh, kh, block_size, qo[head], ko[head]).numpy()
            qb = cluster_boundaries(q_labels[head], qo[head]).numpy()
            kb = cluster_boundaries(k_labels[head], ko[head]).numpy()
            if not np.isfinite(before).all() or not np.isfinite(after).all():
                raise ValueError(f"Nonfinite heatmap scores: step={step}, layer={layer}, head={head}")
            stem = target / f"head_{head:02d}"
            np.savez_compressed(str(stem) + ".npz", before=before, after=after,
                                q_order=qo[head].numpy(), k_order=ko[head].numpy(),
                                q_labels=q_labels[head].cpu().numpy(), k_labels=k_labels[head].cpu().numpy(),
                                q_boundaries=qb, k_boundaries=kb, grid=np.asarray(grid),
                                block_size=np.asarray(block_size))
            limit = max(float(np.abs(before).max()), float(np.abs(after).max()), 1e-12)
            fig, axes = self.plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
            for ax, scores, title in zip(axes, (before, after),
                                         ("Original token order", "Q/K sorted by cluster label")):
                im = ax.imshow(scores, origin="upper", cmap="RdBu_r", vmin=-limit, vmax=limit)
                ax.set_title(title)
                ax.set_xlabel("K latent frame" if ax is axes[0] else "K sorted token block")
                ax.set_ylabel("Q latent frame" if ax is axes[0] else "Q sorted token block")
                ticks = np.unique(np.linspace(0, frames - 1, min(frames, 6), dtype=int))
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
            # Quantize boundaries to pooled bins for legibility; exact token
            # offsets, including empty-cluster omissions, are saved in the NPZ.
            for boundary in np.unique(np.rint(qb[1:-1] / block_size)):
                if 0 < boundary < frames:
                    axes[1].axhline(boundary - 0.5, color="black", alpha=0.25, linewidth=0.4)
            for boundary in np.unique(np.rint(kb[1:-1] / block_size)):
                if 0 < boundary < frames:
                    axes[1].axvline(boundary - 0.5, color="black", alpha=0.25, linewidth=0.4)
            fig.colorbar(im, ax=axes, label="Mean pre-softmax QK score (shared scale)", shrink=0.8)
            fig.suptitle(f"Step {step} | Layer {layer} | Head {head} | {branch}\n"
                         f"{block_size} tokens/block; sorted blocks mix original frames")
            try:
                fig.savefig(str(stem) + ".png", dpi=150)
            finally:
                self.plt.close(fig)
        self.done.add(key)
        self.records.append(dict(layer=layer, step=step, branch=branch,
                                 heads=q.shape[1], grid=list(grid), path=str(target.relative_to(self.output_dir))))
        self._manifest()
        print(f"[Heatmaps] saved {q.shape[1]} head comparisons to {target}", flush=True)

    def finish(self):
        missing = [(layer, branch) for layer in self.layers for branch in sorted(self.branches)
                   if (layer, branch) not in self.done]
        if missing:
            print(f"[Heatmaps] WARNING: missing {missing}; requested step may be dense, skipped, or out of range", flush=True)
        self._manifest()
