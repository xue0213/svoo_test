# Ascend 910C: Wan2.1 T2V 1.3B

This path uses `wan_npu_inference.py` and does not import SVOO's CUDA,
Triton, FlashAttention, or FlashInfer modules. Scope: Wan2.1 T2V 1.3B.
HunyuanVideo and image-to-video entrypoints are not migrated.

Target environment supplied by the user: CANN 8.5.0.alpha002,
PyTorch 2.7.1+CPU, torch_npu 2.7.1.post2. Use the existing vendor-matched
installation. These versions have not been validated on a 910C here.
The reported x86_64 and aarch64 architecture values conflict; use
`uname -m` on the actual inference host before choosing any wheel.

## Environment

Run all commands on the Linux Ascend host, from the project root.
Source your installed CANN environment script first (commonly
`/usr/local/Ascend/ascend-toolkit/set_env.sh`; installation paths vary).
Do not run `scripts/build_env.sh`: it installs NVIDIA CUDA dependencies.

```bash
uname -m
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__); print(torch.npu.is_available())"

# Install only the model/runtime dependencies, retaining the existing torch stack.
python -m pip install diffusers==0.36.0 transformers==4.51.3 accelerate==1.7.0 \
  numpy pillow sentencepiece protobuf ftfy imageio imageio-ffmpeg "huggingface_hub>=0.34,<1"
```

Run from the source checkout; `pip install -e .` is unnecessary for this path.
Check `python -m pip check` afterward for conflicts in your existing environment.
If your container already has a working vendor-modified Diffusers stack, use a
separate environment before changing it.

## Operator and routing tests

These tests require no weights and compare attention output with CPU SDPA.

```bash
SVOO_TEST_DEVICE=npu:0 python -m unittest discover -s tests -p test_npu_attention.py -v
```

If an NPU operator reports unsupported dtype/shape or an ACL error, save the
full traceback with `npu-smi info` and package versions. Do not treat CPU test
success as evidence that the NPU operators are supported by your CANN build.

## Weights

Use the **Diffusers** weights, rather than the original Wan checkpoint layout:

```bash
export MODEL_PATH=/path/to/models/Wan2.1-T2V-1.3B-Diffusers
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir "$MODEL_PATH"
```

## First hardware smoke run

Use dense attention to isolate model/device issues. Two steps and 9 frames
are a connectivity test, not a video quality setting.

```bash
python wan_npu_inference.py --model_id "$MODEL_PATH" --device npu:0 \
  --attention dense --height 256 --width 256 --num_frames 9 \
  --num_inference_steps 2 --output_file result/npu/dense-smoke.mp4
```

Then exercise sparse routing from the first step. Smaller cluster counts
make this smoke test faster:

```bash
python wan_npu_inference.py --model_id "$MODEL_PATH" --device npu:0 \
  --attention svoo --height 256 --width 256 --num_frames 9 \
  --num_inference_steps 2 --first_times_fp 0 --first_layers_fp 0 \
  --num_q_centroids 16 --num_k_centroids 32 \
  --sparsity_csv_path sparsity_profiles/sparsity_wan_1.3B_t2v.csv \
  --output_file result/npu/svoo-smoke.mp4
```

## Normal generation

```bash
python wan_npu_inference.py --model_id "$MODEL_PATH" --device npu:0 \
  --attention svoo --prompt_file data/example/1/prompt.txt \
  --height 720 --width 1280 --num_frames 81 --num_inference_steps 50 \
  --sparsity_csv_path sparsity_profiles/sparsity_wan_1.3B_t2v.csv \
  --output_file result/npu/wan13-svoo.mp4
```

Add `--cpu_offload` to move inactive model components to CPU when memory
is insufficient. This needs enough host RAM and compatible Accelerate NPU
hooks. To reproduce the original size, use 720/1280/81; memory usage on NPU
has not been measured, so the README's CUDA memory numbers are not an NPU
guarantee.

## What this implementation preserves

- Alternating Q/K co-clustering in normalized attention-profile space.
- Key-cluster-size-weighted top-p block selection and minimum cluster ratio.
- Step/layer/head CSV lookup, clipped to 0.15--0.20 by default.
- Dense warmup by layer/step, and cached clustering reuse.
- Original token ordering, BF16 transformer/text encoder/VAE, backend global random generator.

Clustering follows the original BF16/FP32 dtype boundaries with chunked PyTorch operations; it is not bitwise equivalent
to CUDA/BF16 reductions. Selected key/value tokens are gathered for each query
cluster and evaluated by `torch_npu.npu_fusion_attention` in BNSD layout.
Routing metadata moves to CPU, while Q/K/V stay on NPU. This is a functional
reference port, with Python loops and many operator launches. It does **not**
promise the original CUDA sparse speedup. EAR compensation is not implemented.

Offline profiles were generated for the original setup and are reused here;
quality/speed at different resolutions needs evaluation. CPU tests validate
routing and interface behavior, not full pretrained generation.

## Algorithm-alignment audit

The first NPU reference version changed centroid precision, capped cluster
counts, rewrote weighted softmax, and changed VAE precision and generation
defaults. It must not be treated as equivalent to the original implementation.
Those deviations have now been removed:

- CUDA and NPU call the same original `weighted_softmax` and
  `identify_dynamic_map` functions in `svoo/routing.py`.
- Centroids retain the input BF16 dtype; updates accumulate in FP32 then cast
  back, with INT32 counts and the original empty-cluster behavior.
- Centroid profiles use the original BF16 matmul followed by FP32 L2 norm;
  token profiles and profile-distance evaluation use FP32.
- Sampling remains with replacement; cluster counts are never capped by N.
- `top_p=1` still executes clustering, preserving the random-number consumption.
- Default dimensions/frames/steps are 720/1280/81/50, both clustering iteration
  counts are 2, the canonical CSV is enabled, and the negative prompt and BF16
  VAE match the original Wan 1.3B script/entrypoint.
- Warmup follows the original scheduler timestep threshold and strict `>`
  comparison. Reuse follows the original step schedule.
- Q/K/V are ordered with the original label argsort before attention. Tiling
  is optional and disabled by default. Noise uses the backend global generator,
  matching the original entrypoint's sampling approach.

This restores the source-level policy and dtype boundaries; it is **not**
evidence of identical CUDA/NPU intermediate values or generated videos.
Backend random streams, FP32/TF32 dot products, reduction order and sort ties
can differ. Small numerical differences can change a nearest-cluster or top-p
decision, which must be detected rather than dismissed as harmless.

### Original CUDA versus NPU comparison

On a working original CUDA SVOO host, export Q/K/V, initial centroid indices,
cluster labels/counts/centroids, block masks and original FlashInfer output:

```bash
python scripts/validate_npu_parity.py --backend cuda --reference cuda-reference.pt
```

Copy the reference file to the 910C host, then run:

```bash
python scripts/validate_npu_parity.py --backend npu --reference cuda-reference.pt
```

Labels, counts and masks must match **exactly**. Centroids and attention outputs
are compared at the explicitly reported `--atol`/`--rtol` (default 0.02).
For a bitwise requirement, use `--atol 0 --rtol 0`; it is not guaranteed across
different hardware. The attention check uses frozen original CUDA routing to
separate routing mismatch from attention-operator error. Any mismatch fails
the command. One passing fixture does not prove full model equivalence;
representative per-step/per-layer real Q/K/V and final latent/video comparisons
are still needed. CUDA/NPU parity has **not** been run in this workspace.

## Local validation

Thirteen tests passed in both FP32 and BF16 CPU modes, including masked-attention
equivalence, clustering/chunking, warmup/reuse, the native Wan processor
comparison, and a tiny Wan transformer forward in dense and sparse modes.
Local versions: PyTorch 2.6.0+cpu, Diffusers 0.36.0, Transformers 4.51.3,
Accelerate 1.7.0. This is not validation of the supplied 910C/CANN 8.5 stack.

```bash
SVOO_TEST_DTYPE=bf16 python -m unittest discover -s tests -p test_npu_attention.py -v
```

## References

- [Ascend fusion-attention API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_fusion_attention.md)
- [Diffusers 0.36.0 Wan implementation](https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/transformers/transformer_wan.py)
