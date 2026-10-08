"""Wan T2V inference on Ascend NPU. Requires a matched CANN/torch/torch_npu stack."""
import argparse
import math
from pathlib import Path
from copy import deepcopy


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_id", required=True, help="Local Diffusers model directory or Hugging Face ID")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--attention", choices=("dense", "svoo"), default="svoo")
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt_file", type=Path, default=Path("data/example/1/prompt.txt"))
    p.add_argument("--negative_prompt", default='Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards')
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--output_file", default="result/npu/wan.mp4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heatmap_dir", type=Path, help="Enable pooled QK before/after clustering plots")
    p.add_argument("--heatmap_seed", type=int, default=0, help="Independent seed for selecting seven layers")
    p.add_argument("--heatmap_branch", choices=("positive", "negative", "both"), default="positive")
    p.add_argument("--heatmap_step", type=int, help="One-based diffusion step; default first sparse visit per layer")
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--vae_tiling", action="store_true")
    p.add_argument("--num_q_centroids", type=int, default=256)
    p.add_argument("--num_k_centroids", type=int, default=1024)
    p.add_argument("--kmeans_iters", type=int, default=None, help="Override both init and step iterations")
    p.add_argument("--kmeans_iter_init", type=int, default=2)
    p.add_argument("--kmeans_iter_step", type=int, default=2)
    p.add_argument("--cluster_chunk", type=int, default=256)
    p.add_argument("--top_p_kmeans", type=float, default=0.9)
    p.add_argument("--min_kc_ratio", type=float, default=0.1)
    p.add_argument("--first_times_fp", type=float, default=0.2)
    p.add_argument("--first_layers_fp", type=float, default=0.03)
    p.add_argument("--start_reuse_step", type=int, default=11)
    p.add_argument("--reuse_interval", type=int, default=20)
    p.add_argument("--sparsity_csv_path", type=Path, default=Path(__file__).parent / "sparsity_profiles/sparsity_wan_1.3B_t2v.csv", help="Optional Step,Layer,Head,Sparsity CSV")
    p.add_argument("--dynamic_min_kc_ratio_min", type=float, default=0.15)
    p.add_argument("--dynamic_min_kc_ratio_max", type=float, default=0.20)
    a = p.parse_args()
    if a.kmeans_iters is not None:
        a.kmeans_iter_init = a.kmeans_iter_step = a.kmeans_iters
    if not a.device.startswith("npu:") or not a.device[4:].isdigit():
        p.error("--device must be npu:<index>")
    if a.height < 16 or a.width < 16 or a.height % 16 or a.width % 16:
        p.error("height and width must be positive multiples of 16")
    if a.num_frames < 1 or (a.num_frames - 1) % 4:
        p.error("num_frames must be 4*n+1")
    if min(a.num_inference_steps, a.num_q_centroids, a.num_k_centroids,
           a.kmeans_iter_init, a.kmeans_iter_step, a.cluster_chunk, a.reuse_interval, a.start_reuse_step) < 1:
        p.error("Steps, cluster counts, iterations, chunk and reuse values must be positive")
    if not (0 < a.top_p_kmeans <= 1 and 0 <= a.min_kc_ratio <= 1):
        p.error("top_p must be in (0,1]; min_kc_ratio must be in [0,1]")
    if not (0 <= a.first_times_fp <= 1 and 0 <= a.first_layers_fp <= 1):
        p.error("Warmup fractions must be in [0,1]")
    if not 0 <= a.dynamic_min_kc_ratio_min <= a.dynamic_min_kc_ratio_max <= 1:
        p.error("Profile clipping bounds must satisfy 0 <= min <= max <= 1")
    if a.heatmap_dir is not None:
        if a.attention != "svoo":
            p.error("Heatmaps require --attention svoo to capture actual clustering")
        if a.heatmap_step is not None and not 1 <= a.heatmap_step <= a.num_inference_steps:
            p.error("--heatmap_step must be within the inference step range")
    return a


def main():
    args = parse_args()
    import random
    import numpy as np
    import torch
    try:
        import torch_npu
    except ImportError as e:
        raise RuntimeError("Install torch_npu matching PyTorch and CANN on the Ascend host; do not run build_env.sh") from e
    if not torch.npu.is_available():
        raise RuntimeError("No NPU available. Check CANN environment and npu-smi info.")
    torch.npu.set_device(args.device)
    # Official transfer_to_npu also patches CUDA SDPA calls. Avoid it here:
    # all Wan attention is routed explicitly and no CUDA extension is loaded.
    from diffusers import WanPipeline
    from diffusers.utils import export_to_video
    from svoo.models.wan.npu_attention import install_processors, load_profile

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    print(f"torch={torch.__version__} torch_npu={torch_npu.__version__} device={args.device}")
    print(f"Attention={args.attention}. SVOO uses gathered clusters; this is a reference implementation.")
    if args.prompt is None:
        args.prompt = args.prompt_file.read_text(encoding="utf-8").rstrip("\n")
    profile = load_profile(args.sparsity_csv_path)
    # Match the original entrypoint: all loaded components, including VAE, use BF16.
    pipe = WanPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    heatmaps = None
    if args.heatmap_dir is not None:
        from svoo.attention_heatmaps import AttentionHeatmaps
        heatmaps = AttentionHeatmaps(
            args.heatmap_dir, pipe.transformer.config.num_layers, seed=args.heatmap_seed,
            branch=args.heatmap_branch, step=args.heatmap_step,
            metadata=dict(prompt=args.prompt, negative_prompt=args.negative_prompt,
                          model_id=args.model_id, generation_seed=args.seed,
                          torch_version=str(torch.__version__), torch_npu_version=str(torch_npu.__version__),
                          height=args.height, width=args.width, num_frames=args.num_frames,
                          inference_steps=args.num_inference_steps, qc=args.num_q_centroids,
                          kc=args.num_k_centroids, kmeans_iter_init=args.kmeans_iter_init,
                          kmeans_iter_step=args.kmeans_iter_step, first_times_fp=args.first_times_fp,
                          first_layers_fp=args.first_layers_fp, start_reuse_step=args.start_reuse_step,
                          reuse_interval=args.reuse_interval,
                          sparsity_csv_path=str(args.sparsity_csv_path), top_p=args.top_p_kmeans,
                          pooling_note="latent frames, not decoded video frames"),
        )
    scheduler = deepcopy(pipe.scheduler)
    scheduler.set_timesteps(args.num_inference_steps)
    warmup_steps = math.floor(args.first_times_fp * args.num_inference_steps)
    fp_timestep = float(scheduler.timesteps[warmup_steps - 1] - 1) if warmup_steps else 1001
    processors = install_processors(
        pipe, args.attention, qc=args.num_q_centroids, kc=args.num_k_centroids,
        iters_init=args.kmeans_iter_init, iters_step=args.kmeans_iter_step, chunk=args.cluster_chunk,
        top_p=args.top_p_kmeans, min_ratio=args.min_kc_ratio, profile=profile,
        ratio_min=args.dynamic_min_kc_ratio_min, ratio_max=args.dynamic_min_kc_ratio_max,
        fp_steps=warmup_steps, fp_timestep=fp_timestep,
        fp_layers=math.floor(args.first_layers_fp * pipe.transformer.config.num_layers),
        reuse_start=args.start_reuse_step, reuse_interval=args.reuse_interval,
        heatmaps=heatmaps,
    )
    def capture_timestep(module, positional, keyword):
        timestep = keyword.get("timestep")
        if timestep is None and len(positional) > 1:
            timestep = positional[1]
        if timestep is None:
            raise RuntimeError("Wan transformer did not receive scheduler timestep")
        value = float(timestep.flatten()[0])
        for processor in processors:
            processor.current_timestep = value
        if heatmaps is not None:
            hidden = keyword.get("hidden_states")
            if hidden is None:
                hidden = positional[0]
            grid = tuple(int(s // p) for s, p in zip(hidden.shape[-3:], module.config.patch_size))
            for processor in processors:
                processor.token_grid = grid
    pipe.transformer.register_forward_pre_hook(capture_timestep, with_kwargs=True)
    if getattr(pipe, "transformer_2", None) is not None:
        pipe.transformer_2.register_forward_pre_hook(capture_timestep, with_kwargs=True)
    if args.vae_tiling:
        pipe.vae.enable_tiling()
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=int(args.device.split(":")[1]), device="npu")
    else:
        pipe.to(args.device)

    def on_step_end(pipeline, step, timestep, callback_kwargs):
        for processor in processors:
            processor.step = step + 2
        return callback_kwargs

    kwargs = dict(
        prompt=args.prompt, negative_prompt=args.negative_prompt,
        height=args.height, width=args.width, num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps, guidance_scale=5.0,
        callback_on_step_end=on_step_end,
    )
    if getattr(pipe, "transformer_2", None) is not None:
        kwargs["guidance_scale_2"] = 3.0
    with torch.inference_mode():
        frames = pipe(**kwargs).frames[0]
    if heatmaps is not None:
        heatmaps.finish()
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output), fps=16)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
