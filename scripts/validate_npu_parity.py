"""Compare the original CUDA SVOO kernels and the NPU port on identical inputs.

CUDA export requires the original SVOO/FlashInfer build. NPU checking only
needs torch_npu. Matching seeds alone is insufficient: Q/K/V and centroid
initialization indices are explicitly serialized.
"""
import argparse
from pathlib import Path
import sys

# Allow execution from scripts/ without installing the CUDA package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("cuda", "npu"), required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--tokens", type=int, default=129)
    p.add_argument("--qc", type=int, default=16)
    p.add_argument("--kc", type=int, default=32)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--min_ratio", type=float, default=0.15)
    p.add_argument("--atol", type=float, default=0.02)
    p.add_argument("--rtol", type=float, default=0.02)
    args = p.parse_args()
    import torch
    if args.backend == "npu":
        import torch_npu
    device = args.device or args.backend + ":0"
    if torch.device(device).type != args.backend:
        raise ValueError("Device must match the selected CUDA/NPU backend")
    if args.backend == "cuda":
        from unittest.mock import patch
        from svoo.co_clustering import co_cluster_tokens, dynamic_block_sparse_fwd_flashinfer
        from svoo.routing import identify_dynamic_map
        from svoo.kernels.triton.permute import (
            permute_tensor_by_labels_triton, apply_inverse_permutation_triton,
        )
        generator = torch.Generator(device="cpu").manual_seed(123)
        q, k, v = [torch.randn(1, args.heads, args.tokens, 128, generator=generator,
                               dtype=torch.bfloat16) for _ in range(3)]
        qi = torch.randint(args.tokens, (args.heads, args.qc), generator=generator)
        ki = torch.randint(args.tokens, (args.heads, args.kc), generator=generator)
        params = dict(qc=args.qc, kc=args.kc, iters=args.iters,
                      top_p=args.top_p, min_ratio=args.min_ratio)
        fixture = dict(q=q, k=k, v=v, q_indices=qi, k_indices=ki,
                       params=params, backend="cuda")
        q, k, v = (x.to(device) for x in (q, k, v))
        with torch.inference_mode(), patch(
            "torch.randint", side_effect=[qi.to(device), ki.to(device)]
        ) as sampling:
            result = co_cluster_tokens(q.flatten(0, 1), k.flatten(0, 1),
                                       args.qc, args.kc, max_iters=args.iters)
            if sampling.call_count != 2:
                raise RuntimeError("Original co-clustering sampling contract changed")
        ql, qcent, qs, kl, kcent, ks = result
        mask = identify_dynamic_map(
            qcent[None], kcent[None], qs[None], ks[None],
            args.top_p, args.min_ratio,
        )[0]
        qp, order = permute_tensor_by_labels_triton(q, ql, 2)
        kp, korder = permute_tensor_by_labels_triton(k, kl, 2)
        vp, _ = permute_tensor_by_labels_triton(v, kl, 2, sorted_indices=korder)
        op = dynamic_block_sparse_fwd_flashinfer(
            qp, kp, vp, mask[None], qs[None], ks[None], is_cpu=False,
        )
        output = apply_inverse_permutation_triton(op, order, 2)
        fixture["results"] = {
            name: tensor.cpu() for name, tensor in zip(
                ("ql", "qcent", "qs", "kl", "kcent", "ks"), result,
            )
        }
        fixture["results"].update(mask=mask.cpu(), output=output.cpu())
        fixture["torch_version"] = str(torch.__version__)
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fixture, args.reference)
        print(f"Original CUDA reference saved: {args.reference}")
        return

    from svoo.models.wan.npu_attention import co_cluster, select_blocks, gathered_sparse_attention
    fixture = torch.load(args.reference, map_location="cpu", weights_only=True)
    if fixture.get("backend") != "cuda":
        raise ValueError("An original CUDA export is required; CPU self-comparison is not parity")
    q, k, v = (fixture[name].to(device) for name in ("q", "k", "v"))
    params, expected = fixture["params"], fixture["results"]
    with torch.inference_mode():
        result = co_cluster(
            q.flatten(0, 1), k.flatten(0, 1), params["qc"], params["kc"],
            iters=params["iters"], q_indices=fixture["q_indices"], k_indices=fixture["k_indices"],
        )
        ql, qcent, qs, kl, kcent, ks = result
        mask = select_blocks(qcent, kcent, ks, params["top_p"], params["min_ratio"])
        # Compare attention operators with frozen ORIGINAL routing, so a routing
        # mismatch cannot be hidden by a permissive attention tolerance.
        output = gathered_sparse_attention(
            q, k, v, expected["ql"].to(device), expected["kl"].to(device),
            expected["mask"].to(device),
        )
    failed = []
    values = dict(zip(("ql", "qcent", "qs", "kl", "kcent", "ks"), result))
    values.update(mask=mask, output=output)
    for name, tensor in values.items():
        try:
            if name in ("ql", "kl", "qs", "ks", "mask"):
                torch.testing.assert_close(tensor.cpu(), expected[name], atol=0, rtol=0)
            else:
                torch.testing.assert_close(tensor.cpu().float(), expected[name].float(),
                                           atol=args.atol, rtol=args.rtol)
            print(f"PASS {name}")
        except AssertionError as e:
            failed.append(name)
            print(f"FAIL {name}: {e}")
    if failed:
        raise SystemExit("CUDA/NPU parity failed: " + ", ".join(failed))
    print("Fixed-input routing is identical for this fixture; attention meets the stated tolerance.")
    print("This does not establish bitwise or full pretrained-video equivalence.")


if __name__ == "__main__":
    main()
