"""Inspect the original BF16 top-p boundary without changing routing."""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from svoo.models.wan.npu_attention import select_blocks
from svoo.routing import weighted_softmax


def inspect(device):
    print(f"\nDEVICE={device}", flush=True)
    centers = torch.zeros(1, 3, 128, dtype=torch.bfloat16, device=device)
    counts = torch.ones(1, 3, dtype=torch.int32, device=device)
    scores = centers[:, :1] @ centers.transpose(-2, -1) / (128 ** 0.5)
    probs = weighted_softmax(scores, counts.unsqueeze(-2))
    sorted_probs, indices = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(-1)
    threshold = torch.tensor(0.6661, dtype=torch.bfloat16, device=device)
    values = {
        "scores": scores,
        "weighted_probs": probs,
        "sort_indices": indices,
        "cumulative": cumulative,
        "bf16_threshold": threshold,
        "gt_python_scalar": cumulative > 0.6661,
        "gt_bf16_tensor": cumulative > threshold,
        "gt_fp32_scalar": cumulative.float() > 0.6661,
        "selected_mask": select_blocks(centers[:, :1], centers, counts, 0.6661, 0),
    }
    for name, tensor in values.items():
        print(f"{name}: dtype={tensor.dtype}, values={tensor.cpu().tolist()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    print(f"torch={torch.__version__}", flush=True)
    if args.device.startswith("npu"):
        import torch_npu
        print(f"torch_npu={torch_npu.__version__}", flush=True)
        torch.npu.set_device(args.device)
    inspect("cpu")
    if args.device != "cpu":
        inspect(args.device)


if __name__ == "__main__":
    main()
