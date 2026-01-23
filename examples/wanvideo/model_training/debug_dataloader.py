import torch, os, argparse, warnings
import numpy as np
from PIL import Image
from diffsynth.core import UnifiedDataset
from diffsynth.core.data import FilteredYoutubeDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.diffusion.parsers import add_general_config, add_video_size_config
import shutil

try:
    import imageio
except ImportError:
    print("imageio is not installed. Please install it to save videos.")
    imageio = None

def save_video(frames, path, fps=24):
    if not frames:
        return
    if imageio:
        imageio.mimsave(path, [np.array(f) for f in frames], fps=fps)
    else:
        # Fallback to saving frames
        os.makedirs(path + "_frames", exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(os.path.join(path + "_frames", f"{i:04d}.png"))

def wan_parser():
    parser = argparse.ArgumentParser(description="Debug script for dataloader.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    # Add args from train.py that are needed
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary.")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    
    # Debug specific args
    parser.add_argument("--debug_output_path", type=str, default="debug_data", help="Output path for debug data.")
    parser.add_argument("--debug_batches", type=int, default=5, help="Number of batches to debug.")
    return parser

if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    
    # Dataset initialization (copied from train.py)
    if args.dataset_metadata_path and args.dataset_metadata_path.endswith('.json'):
        print(f"Using FilteredYoutubeDataset with metadata: {args.dataset_metadata_path}")
        dataset = FilteredYoutubeDataset(
            base_folder=args.dataset_base_path,
            segment_list_file=args.dataset_metadata_path,
            args=args,
            height=args.height,
            width=args.width,
            sample_frames=args.num_frames,
            max_pixels=args.max_pixels,
            repeat=args.dataset_repeat,
            extract_face_video=True,
        )
    else:
        print(f"Using UnifiedDataset")
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
            special_operator_map={
                "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
                "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            }
        )

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=1)

    if os.path.exists(args.debug_output_path):
        shutil.rmtree(args.debug_output_path)
    os.makedirs(args.debug_output_path, exist_ok=True)

    print(f"Starting debug loop, saving to {args.debug_output_path}...")
    
    for i, data in enumerate(dataloader):
        if i >= args.debug_batches:
            break
        
        batch_dir = os.path.join(args.debug_output_path, f"batch_{i}")
        os.makedirs(batch_dir, exist_ok=True)
        
        print(f"Processing batch {i}...")
        
        # Save prompt
        if "prompt" in data:
            with open(os.path.join(batch_dir, "prompt.txt"), "w") as f:
                f.write(data["prompt"])
        
        # Save video
        if "video" in data:
            save_video(data["video"], os.path.join(batch_dir, "video.mp4"))
            
        # Save control video
        if "control_video" in data:
            save_video(data["control_video"], os.path.join(batch_dir, "control_video.mp4"))
            
        # Save face video
        if "animate_face_video" in data:
            save_video(data["animate_face_video"], os.path.join(batch_dir, "face_video.mp4"))

        # Save reference image
        if "reference_image" in data and len(data["reference_image"]) > 0:
            data["reference_image"][0].save(os.path.join(batch_dir, "reference.jpg"))

    print("Done.")
