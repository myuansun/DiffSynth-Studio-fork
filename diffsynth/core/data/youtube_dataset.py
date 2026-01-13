"""
YouTube Dataset for loading frame sequences from directories.

This module provides dataset classes for loading video data stored as individual
frame images (JPEGs) in directory structures like:

    processed_videos*/
    └── {video_id}/
        └── segment*/
            ├── origin/
            │   └── frame_*.jpg
            └── skeleton/
                └── frame_*.jpg

Classes:
    YoutubeDataset: Base dataset for loading frame sequences
    FilteredYoutubeDataset: YoutubeDataset with allowlist filtering
"""
from __future__ import annotations
import json
import os
import random
import warnings
from typing import Dict, List, Optional, Set, Tuple

import torch
import torchvision
from PIL import Image


class YoutubeDataset(torch.utils.data.Dataset):
    """
    Dataset for loading video data stored as frame sequences in directories.
    
    Directory layout expected:
        processed_video{num}-{num}/
        └── {video_id}/
            └── segment{num}/
                ├── origin/
                │   └── frame_*.jpg
                └── skeleton/
                    └── frame_*.jpg

    __getitem__ returns:
        {
            "prompt": <str>,
            "video": [list of PIL.Image],  # origin frames
            "control_video": [list of PIL.Image],  # skeleton frames
            "reference_image": [list with single PIL.Image]
        }
    """

    def __init__(
        self,
        base_folder: str = None,
        sample_frames: int = 81,
        stride: int = 1,
        landscape_only: bool = True,
        # Prompt sources:
        default_prompt: str = "A single person performing sign language.",
        # Control (pose) options:
        allow_on_the_fly_pose: bool = False,
        device: str = "cuda",
        # Reference image policy:
        reference_strategy: str = "first",  # "first" | "middle" | "random"
        # Resize settings:
        max_pixels: int = 1920*1080,
        height: Optional[int] = None,
        width: Optional[int] = None,
        height_division_factor: int = 16,
        width_division_factor: int = 16,
        # Repeat factor:
        repeat: int = 1,
        # Logging:
        verbose: bool = True,
        # Command-line args (takes precedence if provided):
        args=None,
    ):
        # Extract settings from args if provided
        if args is not None:
            base_folder = getattr(args, 'dataset_base_path', base_folder)
            height = getattr(args, 'height', height)
            width = getattr(args, 'width', width)
            max_pixels = getattr(args, 'max_pixels', max_pixels)
            sample_frames = getattr(args, 'num_frames', sample_frames)
            stride = getattr(args, 'youtube_stride', stride)
            landscape_only = getattr(args, 'youtube_landscape_only', landscape_only)
            reference_strategy = getattr(args, 'youtube_reference_strategy', reference_strategy)
            default_prompt = getattr(args, 'youtube_default_prompt', default_prompt)
            repeat = getattr(args, 'dataset_repeat', repeat)
        
        self.base_folder = base_folder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

        if height is not None and width is not None:
            if verbose:
                print("[YoutubeDataset] Fixed HxW provided; dynamic_resolution=False.")
            self.dynamic_resolution = False
        else:
            if verbose:
                print("[YoutubeDataset] No fixed HxW; dynamic_resolution=True.")
            self.dynamic_resolution = True

        # Dataset config
        self.sample_frames = sample_frames
        self.stride = max(1, stride)
        self.landscape_only = landscape_only
        self.default_prompt = default_prompt
        self.allow_on_the_fly_pose = allow_on_the_fly_pose
        self.device = device
        self.reference_strategy = reference_strategy
        self.repeat = repeat

        # Scan segments
        self.segments: List[Dict] = []
        if base_folder is not None:
            for folder_name in os.listdir(base_folder):
                if folder_name.startswith("processed_video") and "-" in folder_name:
                    folder_path = os.path.join(base_folder, folder_name)
                    if os.path.isdir(folder_path):
                        self._scan_processed_folder(folder_path, verbose=verbose)

        if verbose:
            print(f"[YoutubeDataset] Found {len(self.segments)} segments.")

    def __len__(self) -> int:
        return len(self.segments) * self.repeat

    def __getitem__(self, idx: int) -> Dict:
        seg = self.segments[idx % len(self.segments)]
        origin_frames = seg["origin_frames"]
        origin_path = seg["origin_path"]
        skeleton_path = seg["skeleton_path"]

        # Choose a consecutive snippet with stride
        max_start = len(origin_frames) - (self.sample_frames - 1) * self.stride - 1
        start = random.randint(0, max(0, max_start))
        indices = [start + i * self.stride for i in range(self.sample_frames)]
        names = [origin_frames[min(i, len(origin_frames) - 1)] for i in indices]

        # Load raw PILs
        video_imgs_raw = [self._load_pil(os.path.join(origin_path, f)) for f in names]

        # Determine target (H, W) from first frame
        target_h, target_w = self.get_height_width(video_imgs_raw[0])

        # Resize all streams consistently
        video_imgs = [self.crop_and_resize(img, target_h, target_w) for img in video_imgs_raw]

        control_imgs = []
        for f in names:
            sk_path = os.path.join(skeleton_path, f)
            if os.path.exists(sk_path):
                sk_img = self._load_pil(sk_path, force_rgb=True)
            else:
                # Return black image if skeleton not available
                sk_img = Image.new("RGB", video_imgs_raw[0].size, (0, 0, 0))
            control_imgs.append(self.crop_and_resize(sk_img, target_h, target_w))

        # Reference image
        ref_img_raw = self._load_pil(seg["reference_path"])
        reference_list = [self.crop_and_resize(ref_img_raw, target_h, target_w)]

        return {
            "prompt": seg["prompt"],
            "video": video_imgs,
            "control_video": control_imgs,
            "reference_image": reference_list,
        }

    def _scan_processed_folder(self, folder_path: str, verbose: bool = True):
        """Scan a processed_videos* folder for valid segments."""
        for video_id in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_id)
            if not os.path.isdir(video_path):
                continue
            
            for segment_name in os.listdir(video_path):
                if not segment_name.startswith("segment"):
                    continue

                segment_path = os.path.join(video_path, segment_name)
                origin_path = os.path.join(segment_path, "origin")
                skeleton_path = os.path.join(segment_path, "skeleton")
                
                if not os.path.isdir(origin_path):
                    continue

                origin_frames = sorted([f for f in os.listdir(origin_path) 
                                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                if len(origin_frames) < self.sample_frames:
                    continue

                # Landscape filter
                if self.landscape_only:
                    first_frame = os.path.join(origin_path, origin_frames[0])
                    try:
                        with Image.open(first_frame) as img:
                            w, h = img.size
                            if w <= h:
                                continue
                    except Exception as e:
                        if verbose:
                            print(f"[YoutubeDataset] Skipping unreadable {first_frame}: {e}")
                        continue

                prompt = self._resolve_prompt(segment_path)

                # Reference image selection
                if self.reference_strategy == "first":
                    ref = os.path.join(origin_path, origin_frames[0])
                elif self.reference_strategy == "middle":
                    ref = os.path.join(origin_path, origin_frames[len(origin_frames)//2])
                else:  # random
                    ref = os.path.join(origin_path, random.choice(origin_frames))

                self.segments.append({
                    "video_id": video_id,
                    "segment": segment_name,
                    "origin_path": origin_path,
                    "skeleton_path": skeleton_path,
                    "origin_frames": origin_frames,
                    "prompt": prompt,
                    "reference_path": ref,
                })

    def _resolve_prompt(self, segment_path: str) -> str:
        """Try to load prompt from info.json in segment folder."""
        info_json = os.path.join(segment_path, "info.json")
        if os.path.exists(info_json):
            try:
                with open(info_json, "r", encoding="utf-8") as f:
                    js = json.load(f)
                return js.get("prompt", js.get("caption", self.default_prompt))
            except Exception:
                pass
        return self.default_prompt

    def _load_pil(self, path: str, force_rgb: bool = True) -> Image.Image:
        """Load an image as PIL."""
        img = Image.open(path)
        try:
            img.load()
        except Exception:
            pass
        if force_rgb and img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def crop_and_resize(self, image: Image.Image, target_height: int, target_width: int) -> Image.Image:
        """Center-crop and resize image to target dimensions."""
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def get_height_width(self, image: Image.Image) -> Tuple[int, int]:
        """Calculate target height and width based on image and settings."""
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width


def load_allowed_segments_from_file(segment_list_path: str) -> Set[str]:
    """
    Load a set of allowed segment keys from a segment list file.
    
    Args:
        segment_list_path: Path to the segment list JSON file
    
    Returns:
        Set of segment keys in format "batch_folder/video_id/segment_name"
    """
    with open(segment_list_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'segment_keys' in data:
        return set(data['segment_keys'])
    elif isinstance(data, list):
        return set(data)
    else:
        raise ValueError(f"Invalid segment list format: {type(data)}")


class FilteredYoutubeDataset(YoutubeDataset):
    """
    YoutubeDataset with allowlist filtering based on a segment list file.
    
    Usage:
        # Create the filtered dataset
        dataset = FilteredYoutubeDataset(
            base_folder="/path/to/data",
            segment_list_file="filtered_segments.json",
        )
    """
    
    def __init__(
        self,
        base_folder: str = None,
        segment_list_file: Optional[str] = None,
        allowed_segments: Optional[Set[str]] = None,
        captions_file: Optional[str] = None,  # Load captions from separate file
        args=None,
        **kwargs
    ):
        """
        Args:
            base_folder: Root directory containing processed_video* folders
            segment_list_file: Path to segment list JSON file
            allowed_segments: Set of segment keys to include
            captions_file: Path to captions JSON file (optional, for loading prompts)
            args: Command-line arguments
            **kwargs: Additional arguments passed to YoutubeDataset
        """
        verbose = kwargs.get('verbose', True)
        
        # Extract from args if provided
        if args is not None:
            segment_list_file = getattr(args, 'youtube_segment_list_file', segment_list_file)
            captions_file = getattr(args, 'youtube_captions_file', captions_file)
        
        # Load segment allowlist
        if segment_list_file is not None:
            if verbose:
                print(f"[FilteredYoutubeDataset] Loading segment list from: {segment_list_file}")
            self._allowed_segments = load_allowed_segments_from_file(segment_list_file)
            
            # Also try to load captions from segment list
            with open(segment_list_file, 'r') as f:
                data = json.load(f)
            self._segment_captions = data.get('segment_captions', {})
            
            if verbose:
                print(f"[FilteredYoutubeDataset] Loaded {len(self._allowed_segments)} allowed segments")
                print(f"[FilteredYoutubeDataset] Loaded {len(self._segment_captions)} captions")
        else:
            self._allowed_segments = allowed_segments
            self._segment_captions = {}
        
        # Load additional captions file if provided
        if captions_file is not None and os.path.exists(captions_file):
            if verbose:
                print(f"[FilteredYoutubeDataset] Loading captions from: {captions_file}")
            with open(captions_file, 'r') as f:
                captions_data = json.load(f)
            if 'captions' in captions_data:
                captions_data = captions_data['captions']
            for key, value in captions_data.items():
                if isinstance(value, dict):
                    self._segment_captions[key] = value.get('caption', '')
                else:
                    self._segment_captions[key] = str(value)
        
        super().__init__(base_folder=base_folder, args=args, **kwargs)
        
        if verbose:
            print(f"[FilteredYoutubeDataset] Final segment count: {len(self.segments)}")
    
    def _scan_processed_folder(self, folder_path: str, verbose: bool = True):
        """Override to add allowlist filtering."""
        batch_name = os.path.basename(folder_path)
        
        for video_id in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_id)
            if not os.path.isdir(video_path):
                continue
            
            for segment_name in os.listdir(video_path):
                if not segment_name.startswith("segment"):
                    continue
                
                # Check allowlist
                if self._allowed_segments is not None:
                    segment_key = f"{batch_name}/{video_id}/{segment_name}"
                    if segment_key not in self._allowed_segments:
                        continue
                
                segment_path = os.path.join(video_path, segment_name)
                origin_path = os.path.join(segment_path, "origin")
                skeleton_path = os.path.join(segment_path, "skeleton")
                
                if not os.path.isdir(origin_path):
                    continue

                origin_frames = sorted([f for f in os.listdir(origin_path) 
                                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                if len(origin_frames) < self.sample_frames:
                    continue

                # Landscape filter
                if self.landscape_only:
                    first_frame = os.path.join(origin_path, origin_frames[0])
                    try:
                        with Image.open(first_frame) as img:
                            w, h = img.size
                            if w <= h:
                                continue
                    except Exception as e:
                        if verbose:
                            print(f"[FilteredYoutubeDataset] Skipping: {first_frame}: {e}")
                        continue

                # Get prompt from captions or info.json
                segment_key = f"{batch_name}/{video_id}/{segment_name}"
                if segment_key in self._segment_captions:
                    prompt = self._segment_captions[segment_key]
                else:
                    prompt = self._resolve_prompt(segment_path)

                # Reference image selection
                if self.reference_strategy == "first":
                    ref = os.path.join(origin_path, origin_frames[0])
                elif self.reference_strategy == "middle":
                    ref = os.path.join(origin_path, origin_frames[len(origin_frames)//2])
                else:  # random
                    ref = os.path.join(origin_path, random.choice(origin_frames))

                self.segments.append({
                    "video_id": video_id,
                    "segment": segment_name,
                    "origin_path": origin_path,
                    "skeleton_path": skeleton_path,
                    "origin_frames": origin_frames,
                    "prompt": prompt,
                    "reference_path": ref,
                })
