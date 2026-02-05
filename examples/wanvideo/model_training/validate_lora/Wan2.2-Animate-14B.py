from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


def _default_prompt() -> str:
    return (
        "A studio sign-language video featuring a young East Asian woman with a neat dark-brown chin-length bob haircut "
        "and natural makeup, calm friendly expression. She wears a white blazer and white trousers with a black V-neck "
        "top. Plain dark gray seamless background, soft even studio lighting. Medium waist-up shot, centered framing, "
        "steady camera. She performs continuous clear sign language with precise finger articulation and smooth "
        "transitions; hands always fully visible and never cropped. Realistic motion, sharp detail, no flicker."
    )


def _add_repo_paths():
    script_path = Path(__file__).resolve()
    fork_root = script_path.parents[4]  # DiffSynth-Studio-fork
    repo_root = fork_root.parent  # /home/.../diffsynth
    sys.path.insert(0, str(fork_root))
    sys.path.insert(0, str(repo_root / "Mimicmotion"))
    return fork_root


def _resolve_dwpose_device(dwpose_device: str) -> str:
    if dwpose_device in ("cpu", "cuda"):
        return dwpose_device
    # auto
    if not torch.cuda.is_available():
        return "cpu"
    try:
        import onnxruntime
    except Exception:
        return "cpu"
    providers = set(onnxruntime.get_available_providers())
    return "cuda" if "CUDAExecutionProvider" in providers else "cpu"


def parse_args(fork_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_ckpt", type=Path, default=fork_root / "models" / "step-10000.safetensors")
    parser.add_argument("--motion_video", type=Path, default=fork_root / "test" / "video.mp4")
    parser.add_argument("--ref_image", type=Path, default=fork_root / "test" / "reference_image.png")
    parser.add_argument("--out", type=Path, default=fork_root / "test" / "out_wan22_animate_lora_step10000.mp4")
    parser.add_argument("--dwpose_dir", type=Path, default=fork_root / "models" / "DWPose")
    parser.add_argument("--dwpose_device", type=str, default="cpu", choices=("cpu", "cuda", "auto"))

    parser.add_argument("--prompt", type=str, default=_default_prompt())
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)

    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)

    parser.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--preprocess_only", action="store_true")
    parser.add_argument("--debug_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    fork_root = _add_repo_paths()
    args = parse_args(fork_root)

    from diffsynth.utils.data import save_video
    from diffsynth.utils.wan_animate_preprocess import preprocess_wan_animate_from_video

    if args.debug_dir is None:
        args.debug_dir = args.out.parent / "debug_wan22_animate_inputs"

    animate_inputs = preprocess_wan_animate_from_video(
        args.motion_video,
        args.ref_image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        dwpose_dir=args.dwpose_dir,
        dwpose_device=_resolve_dwpose_device(args.dwpose_device),
        debug_dir=args.debug_dir,
    )

    if args.preprocess_only:
        pose_out = args.debug_dir / "animate_pose_video.mp4"
        face_out = args.debug_dir / "animate_face_video.mp4"
        save_video(animate_inputs.animate_pose_video, str(pose_out), fps=args.fps, quality=args.quality)
        save_video(animate_inputs.animate_face_video, str(face_out), fps=args.fps, quality=args.quality)
        print(f"[OK] Preprocess-only. Wrote: {pose_out}")
        print(f"[OK] Preprocess-only. Wrote: {face_out}")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run with `--device cpu --preprocess_only`, or run on a CUDA machine.")

    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device=args.device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
    )

    if not args.lora_ckpt.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {args.lora_ckpt}")
    pipe.load_lora(pipe.dit, str(args.lora_ckpt), alpha=args.alpha)

    video = pipe(
        prompt=args.prompt,
        seed=args.seed,
        tiled=True,
        input_image=animate_inputs.input_image,
        animate_pose_video=animate_inputs.animate_pose_video,
        animate_face_video=animate_inputs.animate_face_video,
        num_frames=animate_inputs.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
    )
    save_video(video, str(args.out), fps=args.fps, quality=args.quality)
    print(f"[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()
