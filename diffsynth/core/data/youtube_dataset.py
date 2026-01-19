"""
YouTube Dataset for loading frame sequences from directories.

This module provides dataset classes for loading video data stored as individual
frame images (JPEGs) in directory structures like:

    processed_videos*/
    └── {video_id}/
        └── segment*/
            ├── origin/
            │   └── frame_*.jpg
            ├── skeleton/
            │   └── frame_*.jpg
            └── keypoints.npy

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

import numpy as np
import torch
import torchvision
from PIL import Image


def compute_keypoint_bbox(
    keypoints_frame: dict,
    img_width: int,
    img_height: int,
    padding: float = 0.1,
    min_confidence: float = 0.3,
) -> Tuple[int, int, int, int]:
    """
    Compute bounding box from keypoints for a single frame.
    
    Args:
        keypoints_frame: Dict with 'bodies', 'hands', 'faces' keys
        img_width: Original image width
        img_height: Original image height  
        padding: Fraction of bbox size to add as padding
        min_confidence: Minimum confidence to include a keypoint
    
    Returns:
        (x1, y1, x2, y2) bounding box in pixel coordinates
    """
    all_points = []
    
    # Extract body keypoints
    if 'bodies' in keypoints_frame and keypoints_frame['bodies']:
        bodies = keypoints_frame['bodies']
        if isinstance(bodies, dict) and 'candidate' in bodies:
            candidates = bodies['candidate']
            scores = bodies.get('score', np.ones((1, len(candidates))))
            if len(scores.shape) == 2:
                scores = scores[0]  # Take first person's scores
            for i, pt in enumerate(candidates):
                if i < len(scores) and scores[i] > min_confidence:
                    all_points.append(pt)
    
    # Extract hand keypoints
    if 'hands' in keypoints_frame and keypoints_frame['hands'] is not None:
        hands = keypoints_frame['hands']
        hands_score = keypoints_frame.get('hands_score')
        for hand_idx in range(len(hands)):
            hand_pts = hands[hand_idx]
            for pt_idx, pt in enumerate(hand_pts):
                if hands_score is not None and hands_score[hand_idx, pt_idx] > min_confidence:
                    all_points.append(pt)
                elif hands_score is None:
                    all_points.append(pt)
    
    # Extract face keypoints (less weight - just for inclusion)
    if 'faces' in keypoints_frame and keypoints_frame['faces'] is not None:
        faces = keypoints_frame['faces']
        if len(faces) > 0:
            # Just use face center, not all 68 points
            face_pts = faces[0]
            face_center = np.mean(face_pts, axis=0)
            all_points.append(face_center)
    
    if not all_points:
        # No valid keypoints - return full image
        return (0, 0, img_width, img_height)
    
    all_points = np.array(all_points)
    
    # Convert normalized coords to pixels
    x_coords = all_points[:, 0] * img_width
    y_coords = all_points[:, 1] * img_height
    
    # Compute bbox
    x1, x2 = x_coords.min(), x_coords.max()
    y1, y2 = y_coords.min(), y_coords.max()
    
    # Add padding
    w, h = x2 - x1, y2 - y1
    pad_x = w * padding
    pad_y = h * padding
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_width, x2 + pad_x)
    y2 = min(img_height, y2 + pad_y)
    
    x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
    if x2_i <= x1_i:
        x1_i = max(0, min(x1_i, img_width - 1))
        x2_i = min(img_width, x1_i + 1)
    if y2_i <= y1_i:
        y1_i = max(0, min(y1_i, img_height - 1))
        y2_i = min(img_height, y1_i + 1)
    return (x1_i, y1_i, x2_i, y2_i)


def compute_adaptive_crop_box(
    bbox: Tuple[int, int, int, int],
    img_width: int,
    img_height: int,
    target_aspect: float,  # height / width
) -> Tuple[int, int, int, int]:
    """
    Expand bbox to match target aspect ratio while staying within image bounds.
    
    Args:
        bbox: (x1, y1, x2, y2) keypoint bounding box
        img_width: Original image width
        img_height: Original image height
        target_aspect: Target aspect ratio (height / width)
    
    Returns:
        (x1, y1, x2, y2) adjusted crop box
    """
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    bbox_cx = (x1 + x2) / 2
    bbox_cy = (y1 + y2) / 2
    
    current_aspect = bbox_h / max(bbox_w, 1)
    
    if current_aspect < target_aspect:
        # Need more height
        new_h = bbox_w * target_aspect
        new_w = bbox_w
    else:
        # Need more width
        new_w = bbox_h / target_aspect
        new_h = bbox_h
    
    # Expand to fit within image while keeping center
    # Check if we need to expand further to stay within bounds
    half_w = new_w / 2
    half_h = new_h / 2
    
    # Adjust center if crop would go out of bounds
    cx = max(half_w, min(img_width - half_w, bbox_cx))
    cy = max(half_h, min(img_height - half_h, bbox_cy))
    
    # If crop is larger than image, constrain to image size
    if new_w > img_width:
        new_w = img_width
        cx = img_width / 2
    if new_h > img_height:
        new_h = img_height
        cy = img_height / 2
    
    # Final bbox
    x1 = int(max(0, cx - new_w / 2))
    y1 = int(max(0, cy - new_h / 2))
    x2 = int(min(img_width, cx + new_w / 2))
    y2 = int(min(img_height, cy + new_h / 2))
    
    return (x1, y1, x2, y2)


class YoutubeDataset(torch.utils.data.Dataset):
    """
    Dataset for loading video data stored as frame sequences in directories.
    
    Directory layout expected:
        processed_video{num}-{num}/
        └── {video_id}/
            └── segment{num}/
                ├── origin/
                │   └── frame_*.jpg
                ├── skeleton/
                │   └── frame_*.jpg
                └── keypoints.npy (optional, for keypoint-aware cropping)

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
        # Keypoint-aware cropping:
        use_keypoint_crop: bool = True,
        keypoint_padding: float = 0.15,
        # Face extraction for animate_face_video:
        extract_face_video: bool = False,
        face_crop_size: int = 512,
        face_padding: float = 0.5,
        face_upper_shift: float = 0.25,
        # Repeat factor:
        repeat: int = 1,
        # Smart repeat: expand segments into unique windows instead of flat repeat
        smart_repeat: bool = False,
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
            smart_repeat = getattr(args, 'youtube_smart_repeat', smart_repeat)
            use_keypoint_crop = getattr(args, 'youtube_keypoint_crop', use_keypoint_crop)
            keypoint_padding = getattr(args, 'youtube_keypoint_padding', keypoint_padding)
            extract_face_video = getattr(args, 'youtube_extract_face_video', extract_face_video)
            face_crop_size = getattr(args, 'youtube_face_crop_size', face_crop_size)
            face_padding = getattr(args, 'youtube_face_padding', face_padding)
            face_upper_shift = getattr(args, 'youtube_face_upper_shift', face_upper_shift)
        
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
        self.smart_repeat = smart_repeat
        
        # Keypoint cropping config
        self.use_keypoint_crop = use_keypoint_crop
        self.keypoint_padding = keypoint_padding
        
        # Face extraction config
        self.extract_face_video = extract_face_video
        self.face_crop_size = face_crop_size
        self.face_padding = face_padding
        self.face_upper_shift = face_upper_shift

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
            if use_keypoint_crop:
                kp_count = sum(1 for s in self.segments if s.get("keypoints_path"))
                print(f"[YoutubeDataset] {kp_count}/{len(self.segments)} have keypoints for adaptive cropping.")
        
        # Build samples list for smart_repeat mode
        self.samples: List[Tuple[int, int]] = []  # (segment_idx, start_frame)
        if self.smart_repeat:
            self._build_smart_samples(verbose=verbose)

    def _build_smart_samples(self, verbose: bool = True):
        """
        Expand segments into unique (segment_idx, start_frame) pairs.
        Each unique window becomes one sample, so longer segments contribute more.
        """
        self.samples = []
        for seg_idx, seg in enumerate(self.segments):
            num_frames = len(seg['origin_frames'])
            # Calculate number of unique windows
            # max_start = num_frames - (sample_frames - 1) * stride - 1
            max_start = num_frames - (self.sample_frames - 1) * self.stride - 1
            num_windows = max(1, max_start + 1)
            for start in range(num_windows):
                self.samples.append((seg_idx, start))
        
        if verbose:
            print(f"[YoutubeDataset] Smart repeat: expanded to {len(self.samples)} unique windows.")
            print(f"[YoutubeDataset] Average windows per segment: {len(self.samples) / len(self.segments):.1f}")

    def __len__(self) -> int:
        if self.smart_repeat:
            return len(self.samples) * self.repeat
        return len(self.segments) * self.repeat

    def __getitem__(self, idx: int) -> Dict:
        if self.smart_repeat:
            # Smart repeat mode: use precomputed (segment_idx, start) pairs
            sample_idx = idx % len(self.samples)
            seg_idx, start = self.samples[sample_idx]
            seg = self.segments[seg_idx]
        else:
            # Original mode: random start within segment
            seg = self.segments[idx % len(self.segments)]
            start = None  # Will be computed below
        
        origin_frames = seg["origin_frames"]
        origin_path = seg["origin_path"]
        skeleton_path = seg["skeleton_path"]
        keypoints_path = seg.get("keypoints_path")

        # Choose a consecutive snippet with stride
        if start is None:
            # Original random sampling mode
            max_start = len(origin_frames) - (self.sample_frames - 1) * self.stride - 1
            start = random.randint(0, max(0, max_start))
        
        indices = [start + i * self.stride for i in range(self.sample_frames)]
        names = [origin_frames[min(i, len(origin_frames) - 1)] for i in indices]

        # Load keypoints if available (for both keypoint crop and face extraction)
        keypoints = None
        if (self.use_keypoint_crop or self.extract_face_video) and keypoints_path and os.path.exists(keypoints_path):
            try:
                keypoints = np.load(keypoints_path, allow_pickle=True)
            except Exception:
                keypoints = None

        # Load first frame to determine dimensions
        first_img = self._load_pil(os.path.join(origin_path, names[0]))
        img_width, img_height = first_img.size
        
        # Determine target dimensions
        target_h, target_w = self.get_height_width(first_img)
        target_aspect = target_h / target_w

        # Compute crop boxes for each frame
        crop_boxes = []
        for frame_idx, name in zip(indices, names):
            if keypoints is not None and frame_idx < len(keypoints):
                kp_frame = keypoints[frame_idx]
                bbox = compute_keypoint_bbox(
                    kp_frame, img_width, img_height,
                    padding=self.keypoint_padding
                )
                crop_box = compute_adaptive_crop_box(bbox, img_width, img_height, target_aspect)
            else:
                # Fallback to center crop
                crop_box = self._center_crop_box(img_width, img_height, target_aspect)
            crop_boxes.append(crop_box)

        # Load and crop origin frames
        video_imgs = []
        for name, crop_box in zip(names, crop_boxes):
            img = self._load_pil(os.path.join(origin_path, name))
            img = self._crop_and_resize(img, crop_box, target_h, target_w)
            video_imgs.append(img)

        # Load and crop skeleton frames with same crop boxes
        control_imgs = []
        for name, crop_box in zip(names, crop_boxes):
            sk_path = os.path.join(skeleton_path, name)
            if os.path.exists(sk_path):
                sk_img = self._load_pil(sk_path, force_rgb=True)
            else:
                # Return black image if skeleton not available
                sk_img = Image.new("RGB", (img_width, img_height), (0, 0, 0))
            sk_img = self._crop_and_resize(sk_img, crop_box, target_h, target_w)
            control_imgs.append(sk_img)

        # Reference image - use first frame's crop
        ref_img = self._load_pil(seg["reference_path"])
        ref_crop_box = crop_boxes[0] if crop_boxes else self._center_crop_box(
            ref_img.size[0], ref_img.size[1], target_aspect
        )
        ref_img = self._crop_and_resize(ref_img, ref_crop_box, target_h, target_w)
        reference_list = [ref_img]

        # Extract face crops for animate_face_video if enabled
        face_crops = None
        if self.extract_face_video:
            face_crops = []
            for frame_idx, name in zip(indices, names):
                img = self._load_pil(os.path.join(origin_path, name))
                face_box = self._compute_face_crop_box(
                    keypoints, frame_idx, img.size[0], img.size[1]
                )
                face_crop = self._crop_and_resize(img, face_box, self.face_crop_size, self.face_crop_size)
                face_crops.append(face_crop)

        result = {
            "prompt": seg["prompt"],
            "video": video_imgs,
            "control_video": control_imgs,
            "reference_image": reference_list,
            "animate_pose_video": control_imgs,  # Alias for Animate model
        }
        if face_crops is not None:
            result["animate_face_video"] = face_crops
        return result

    def _center_crop_box(
        self, img_width: int, img_height: int, target_aspect: float
    ) -> Tuple[int, int, int, int]:
        """Compute center crop box for given target aspect ratio."""
        current_aspect = img_height / img_width
        
        if current_aspect > target_aspect:
            # Image is taller - crop height
            new_h = int(img_width * target_aspect)
            y1 = (img_height - new_h) // 2
            return (0, y1, img_width, y1 + new_h)
        else:
            # Image is wider - crop width
            new_w = int(img_height / target_aspect)
            x1 = (img_width - new_w) // 2
            return (x1, 0, x1 + new_w, img_height)

    def _crop_and_resize(
        self,
        image: Image.Image,
        crop_box: Tuple[int, int, int, int],
        target_height: int,
        target_width: int,
    ) -> Image.Image:
        """Crop image using box and resize to target dimensions."""
        x1, y1, x2, y2 = crop_box
        cropped = image.crop((x1, y1, x2, y2))
        resized = torchvision.transforms.functional.resize(
            cropped,
            (target_height, target_width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return resized

    def _compute_face_crop_box(
        self,
        keypoints,
        frame_idx: int,
        img_width: int,
        img_height: int,
    ) -> Tuple[int, int, int, int]:
        """
        Compute a square crop box centered on the face from facial landmarks.
        
        Args:
            keypoints: Loaded keypoints array or None
            frame_idx: Index of the frame in the keypoints array
            img_width: Original image width
            img_height: Original image height
        
        Returns:
            (x1, y1, x2, y2) square crop box in pixel coordinates
        """
        # Default to center square crop
        size = min(img_width, img_height)
        cx, cy = img_width / 2, img_height / 2
        
        # Try to get face landmarks
        if keypoints is not None and frame_idx < len(keypoints):
            kp_frame = keypoints[frame_idx]
            faces = kp_frame.get('faces') if isinstance(kp_frame, dict) else None
            if faces is not None and len(faces) > 0:
                face_pts = faces[0]  # Use first detected face (68, 2)
                
                # Convert normalized coords to pixels
                x_coords = face_pts[:, 0] * img_width
                y_coords = face_pts[:, 1] * img_height
                
                # Compute face bounding box
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                # Add padding
                face_w = x_max - x_min
                face_h = y_max - y_min
                pad_x = face_w * self.face_padding
                pad_y = face_h * self.face_padding
                
                x_min -= pad_x
                y_min -= pad_y
                x_max += pad_x
                y_max += pad_y
                
                # Make square (use larger dimension)
                box_w = x_max - x_min
                box_h = y_max - y_min
                size = max(box_w, box_h)
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                
                # Shift center upward to include forehead/top of head
                # (68-point landmarks only cover jawline to eyebrows)
                cy -= face_h * self.face_upper_shift
        
        # Compute square crop centered at (cx, cy)
        half_size = size / 2
        
        # Clamp center so crop stays within image
        cx = max(half_size, min(img_width - half_size, cx))
        cy = max(half_size, min(img_height - half_size, cy))
        
        x1 = int(max(0, cx - half_size))
        y1 = int(max(0, cy - half_size))
        x2 = int(min(img_width, cx + half_size))
        y2 = int(min(img_height, cy + half_size))
        
        return (x1, y1, x2, y2)

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
                keypoints_path = os.path.join(segment_path, "keypoints.npy")
                
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
                    "segment_path": segment_path,
                    "origin_path": origin_path,
                    "skeleton_path": skeleton_path,
                    "keypoints_path": keypoints_path if os.path.exists(keypoints_path) else None,
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
                keypoints_path = os.path.join(segment_path, "keypoints.npy")
                
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
                    "segment_path": segment_path,
                    "origin_path": origin_path,
                    "skeleton_path": skeleton_path,
                    "keypoints_path": keypoints_path if os.path.exists(keypoints_path) else None,
                    "origin_frames": origin_frames,
                    "prompt": prompt,
                    "reference_path": ref,
                })
