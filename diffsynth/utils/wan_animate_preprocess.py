from __future__ import annotations

import colorsys
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from diffsynth.core.data.youtube_dataset import compute_keypoint_bbox


class WanAnimatePreprocessError(RuntimeError):
    pass


@dataclass(frozen=True)
class WanAnimateInputs:
    input_image: Image.Image
    animate_pose_video: list[Image.Image]
    animate_face_video: list[Image.Image]
    num_frames: int
    crop_box: tuple[int, int, int, int]


def _largest_num_frames_leq(num_frames: int, *, time_division_factor: int = 4, time_division_remainder: int = 1) -> int:
    if num_frames <= 0:
        return 0
    while num_frames > 1 and num_frames % time_division_factor != time_division_remainder:
        num_frames -= 1
    return num_frames


def _center_crop_box(img_width: int, img_height: int, target_aspect: float) -> tuple[int, int, int, int]:
    current_aspect = img_height / img_width
    if current_aspect > target_aspect:
        new_h = int(img_width * target_aspect)
        y1 = (img_height - new_h) // 2
        return (0, y1, img_width, y1 + new_h)
    new_w = int(img_height / target_aspect)
    x1 = (img_width - new_w) // 2
    return (x1, 0, x1 + new_w, img_height)


def _compute_adaptive_crop_box_strict(
    bbox: tuple[int, int, int, int],
    img_width: int,
    img_height: int,
    target_aspect: float,  # height / width
) -> tuple[int, int, int, int]:
    """
    Compute a crop box that:
      - is inside the image
      - (approximately) matches `target_aspect`
      - is centered near the bbox center
      - includes the bbox when possible
    """
    x1, y1, x2, y2 = bbox
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    bbox_cx = (x1 + x2) / 2
    bbox_cy = (y1 + y2) / 2

    # Minimal crop that covers bbox, expanded to match target aspect.
    if bbox_h / bbox_w > target_aspect:
        crop_h = float(bbox_h)
        crop_w = crop_h / target_aspect
    else:
        crop_w = float(bbox_w)
        crop_h = crop_w * target_aspect

    # Clamp to image bounds while preserving aspect ratio.
    if crop_w > img_width:
        crop_w = float(img_width)
        crop_h = crop_w * target_aspect
    if crop_h > img_height:
        crop_h = float(img_height)
        crop_w = crop_h / target_aspect

    crop_w_i = max(1, min(img_width, int(round(crop_w))))
    crop_h_i = max(1, min(img_height, int(round(crop_h))))
    # Re-sync dimensions to aspect using width as the source of truth.
    crop_h_i = max(1, min(img_height, int(round(crop_w_i * target_aspect))))
    if crop_h_i > img_height:
        crop_h_i = img_height
        crop_w_i = max(1, min(img_width, int(round(crop_h_i / target_aspect))))
        crop_h_i = max(1, min(img_height, int(round(crop_w_i * target_aspect))))

    left = int(round(bbox_cx - crop_w_i / 2))
    top = int(round(bbox_cy - crop_h_i / 2))

    # If bbox fits in crop, adjust so bbox is fully inside.
    if crop_w_i >= bbox_w:
        min_left = x2 - crop_w_i
        max_left = x1
        left = max(min_left, min(left, max_left))
    if crop_h_i >= bbox_h:
        min_top = y2 - crop_h_i
        max_top = y1
        top = max(min_top, min(top, max_top))

    left = max(0, min(img_width - crop_w_i, left))
    top = max(0, min(img_height - crop_h_i, top))
    return (int(left), int(top), int(left + crop_w_i), int(top + crop_h_i))


def _crop_and_resize(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    target_height: int,
    target_width: int,
) -> Image.Image:
    x1, y1, x2, y2 = crop_box
    cropped = image.crop((x1, y1, x2, y2))
    return torchvision.transforms.functional.resize(
        cropped,
        (target_height, target_width),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )


def ensure_dwpose_weights(dwpose_dir: Path) -> tuple[Path, Path]:
    """
    Ensure DWPose ONNX weights exist at:
      - {dwpose_dir}/yolox_l.onnx
      - {dwpose_dir}/dw-ll_ucoco_384.onnx
    """
    dwpose_dir.mkdir(parents=True, exist_ok=True)
    det_path = dwpose_dir / "yolox_l.onnx"
    pose_path = dwpose_dir / "dw-ll_ucoco_384.onnx"

    if det_path.exists() and pose_path.exists():
        return det_path, pose_path

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # pragma: no cover
        raise WanAnimatePreprocessError(
            "Missing dependency `huggingface_hub`. Install it in your environment to auto-download DWPose weights."
        ) from e

    cache_dir = dwpose_dir / ".hf_cache"
    # Avoid writing to ~/.cache in sandboxed environments.
    hf_home = dwpose_dir.parent / ".hf_home"
    hub_cache = dwpose_dir.parent / ".hf_cache"
    xet_cache = dwpose_dir.parent / ".hf_xet"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("HF_XET_CACHE", str(xet_cache))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if not det_path.exists():
        src = hf_hub_download(repo_id="yzd-v/DWPose", filename="yolox_l.onnx", cache_dir=str(cache_dir))
        shutil.copyfile(src, det_path)
    if not pose_path.exists():
        src = hf_hub_download(repo_id="yzd-v/DWPose", filename="dw-ll_ucoco_384.onnx", cache_dir=str(cache_dir))
        shutil.copyfile(src, pose_path)
    return det_path, pose_path


def _read_video_first_n_frames_rgb(video_path: Path, n: int) -> list[np.ndarray]:
    import imageio

    reader = imageio.get_reader(str(video_path))
    frames: list[np.ndarray] = []
    try:
        for frame in reader:
            frames.append(frame)
            if len(frames) >= n:
                break
    finally:
        reader.close()
    return frames


def _dwpose_device_arg(device: str) -> object:
    """
    MimicMotion's Wholebody provider selection treats torch.device('cpu') as CUDA.
    Work around by passing the string 'cpu' when on CPU, and torch.device('cuda') on GPU.
    """
    if device == "cpu":
        return "cpu"
    return torch.device(device)


def extract_dwpose_keypoints(
    frames_rgb: list[np.ndarray],
    det_onnx_path: Path,
    pose_onnx_path: Path,
    *,
    device: str,
) -> list[dict]:
    try:
        import onnxruntime  # noqa: F401
    except Exception as e:
        raise WanAnimatePreprocessError(
            "Missing dependency `onnxruntime`. Install it (e.g. `pip install onnxruntime`) to run DWPose."
        ) from e

    from mimicmotion.dwpose.dwpose_detector import DWposeDetector

    detector = DWposeDetector(
        model_det=str(det_onnx_path),
        model_pose=str(pose_onnx_path),
        device=_dwpose_device_arg(device),
    )
    poses: list[dict] = []
    for frame in tqdm(frames_rgb, desc="DWPose"):
        poses.append(detector(frame))
    detector.release_memory()
    return poses


def _compute_fixed_crop_box(
    keypoints: list[dict],
    img_width: int,
    img_height: int,
    *,
    target_height: int,
    target_width: int,
    keypoint_padding: float = 0.15,
) -> tuple[int, int, int, int]:
    target_aspect = target_height / target_width
    bboxes = [
        compute_keypoint_bbox(kp, img_width, img_height, padding=keypoint_padding)
        for kp in keypoints
    ]
    if not bboxes:
        return _center_crop_box(img_width, img_height, target_aspect)

    mean_bbox = np.mean(np.array(bboxes), axis=0)
    fixed_bbox = (int(mean_bbox[0]), int(mean_bbox[1]), int(mean_bbox[2]), int(mean_bbox[3]))
    return _compute_adaptive_crop_box_strict(fixed_bbox, img_width, img_height, target_aspect)


def _compute_face_crop_box(
    keypoints_frame: Optional[dict],
    frame_idx: int,
    img_width: int,
    img_height: int,
    *,
    face_padding: float = 0.5,
    face_upper_shift: float = 0.25,
) -> tuple[int, int, int, int]:
    size = min(img_width, img_height)
    cx, cy = img_width / 2, img_height / 2

    if isinstance(keypoints_frame, dict):
        faces = keypoints_frame.get("faces")
        if faces is not None and len(faces) > 0:
            face_pts = faces[0]
            x_coords = face_pts[:, 0] * img_width
            y_coords = face_pts[:, 1] * img_height
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            face_w = x_max - x_min
            face_h = y_max - y_min
            pad_x = face_w * face_padding
            pad_y = face_h * face_padding
            x_min -= pad_x
            y_min -= pad_y
            x_max += pad_x
            y_max += pad_y

            box_w = x_max - x_min
            box_h = y_max - y_min
            size = max(box_w, box_h)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            cy -= face_h * face_upper_shift

    half_size = size / 2
    cx = max(half_size, min(img_width - half_size, cx))
    cy = max(half_size, min(img_height - half_size, cy))
    x1 = int(max(0, cx - half_size))
    y1 = int(max(0, cy - half_size))
    x2 = int(min(img_width, cx + half_size))
    y2 = int(min(img_height, cy + half_size))
    return (x1, y1, x2, y2)


def _alpha_blend_color(color, alpha: float) -> list[int]:
    return [int(c * alpha) for c in color]


def _draw_bodypose(canvas: np.ndarray, candidate, subset, score) -> np.ndarray:
    h, w, _ = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4
    limb_seq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18], [3, 17], [6, 18],
    ]
    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],
    ]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limb_seq[i]) - 1]
            conf = score[n][np.array(limb_seq[i]) - 1]
            if conf[0] < 0.3 or conf[1] < 0.3:
                continue
            y = candidate[index.astype(int), 0] * float(w)
            x = candidate[index.astype(int), 1] * float(h)
            mx = np.mean(x)
            my = np.mean(y)
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(x[0] - x[1], y[0] - y[1]))
            polygon = cv2.ellipse2Poly((int(my), int(mx)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, _alpha_blend_color(colors[i], float(conf[0] * conf[1])))

    canvas = (canvas * 0.6).astype(np.uint8)
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = float(score[n][i])
            x = int(x * w)
            y = int(y * h)
            cv2.circle(canvas, (int(x), int(y)), 4, _alpha_blend_color(colors[i], conf), thickness=-1)
    return canvas


def _draw_handpose(canvas: np.ndarray, all_hand_peaks, all_hand_scores) -> np.ndarray:
    h, w, _ = canvas.shape
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
        [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20],
    ]

    for peaks, scores in zip(all_hand_peaks, all_hand_scores):
        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
            score = int(scores[e[0]] * scores[e[1]] * 255)
            if x1 > 0.01 and y1 > 0.01 and x2 > 0.01 and y2 > 0.01:
                r, g, b = colorsys.hsv_to_rgb(ie / float(len(edges)), 1.0, 1.0)
                cv2.line(canvas, (x1, y1), (x2, y2), (r * score, g * score, b * score), thickness=2)
        for i, keypoint in enumerate(peaks):
            x, y = keypoint
            x = int(x * w)
            y = int(y * h)
            score = int(scores[i] * 255)
            if x > 0.01 and y > 0.01:
                cv2.circle(canvas, (x, y), 4, (0, 0, score), thickness=-1)
    return canvas


def _draw_facepose(canvas: np.ndarray, all_lmks, all_scores) -> np.ndarray:
    h, w, _ = canvas.shape
    for lmks, scores in zip(all_lmks, all_scores):
        for lmk, score in zip(lmks, scores):
            x, y = lmk
            x = int(x * w)
            y = int(y * h)
            conf = int(score * 255)
            if x > 0.01 and y > 0.01:
                cv2.circle(canvas, (x, y), 3, (conf, conf, conf), thickness=-1)
    return canvas


def draw_pose_image_rgb(pose: dict, height: int, width: int, *, ref_w: int = 2160) -> np.ndarray:
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]

    sz = min(height, width)
    sr = (ref_w / sz) if sz != ref_w else 1
    canvas = np.zeros(shape=(int(height * sr), int(width * sr), 3), dtype=np.uint8)
    canvas = _draw_bodypose(canvas, candidate, subset, score=bodies["score"])
    canvas = _draw_handpose(canvas, hands, pose["hands_score"])
    canvas = _draw_facepose(canvas, faces, pose["faces_score"])
    canvas = cv2.resize(canvas, (width, height))
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas


def preprocess_wan_animate_from_video(
    motion_video_path: Path,
    reference_image_path: Path,
    *,
    height: int,
    width: int,
    num_frames: int = 81,
    dwpose_dir: Path,
    dwpose_device: str = "cpu",
    keypoint_padding: float = 0.15,
    face_crop_size: int = 512,
    face_padding: float = 0.5,
    face_upper_shift: float = 0.25,
    debug_dir: Optional[Path] = None,
) -> WanAnimateInputs:
    if num_frames <= 0:
        raise WanAnimatePreprocessError(f"Invalid num_frames={num_frames}.")
    motion_video_path = Path(motion_video_path)
    reference_image_path = Path(reference_image_path)
    if not motion_video_path.exists():
        raise WanAnimatePreprocessError(f"motion_video_path not found: {motion_video_path}")
    if not reference_image_path.exists():
        raise WanAnimatePreprocessError(f"reference_image_path not found: {reference_image_path}")

    frames_rgb = _read_video_first_n_frames_rgb(motion_video_path, num_frames)
    if len(frames_rgb) < num_frames:
        num_frames = _largest_num_frames_leq(len(frames_rgb))
        frames_rgb = frames_rgb[:num_frames]
    if num_frames < 5:
        raise WanAnimatePreprocessError(
            f"Need at least 5 frames to build Animate inputs (got {num_frames})."
        )

    img_h, img_w = frames_rgb[0].shape[:2]
    det_onnx, pose_onnx = ensure_dwpose_weights(dwpose_dir)
    keypoints = extract_dwpose_keypoints(frames_rgb, det_onnx, pose_onnx, device=dwpose_device)

    crop_box = _compute_fixed_crop_box(
        keypoints,
        img_width=img_w,
        img_height=img_h,
        target_height=height,
        target_width=width,
        keypoint_padding=keypoint_padding,
    )

    pose_frames: list[Image.Image] = []
    face_frames: list[Image.Image] = []
    animate_len = num_frames - 4

    for i in tqdm(range(num_frames), desc="Build Animate inputs"):
        frame_pil = Image.fromarray(frames_rgb[i]).convert("RGB")

        pose_rgb = draw_pose_image_rgb(keypoints[i], img_h, img_w)
        pose_pil = Image.fromarray(pose_rgb).convert("RGB")
        pose_pil = _crop_and_resize(pose_pil, crop_box, height, width)
        pose_frames.append(pose_pil)

        face_box = _compute_face_crop_box(
            keypoints[i],
            i,
            img_width=img_w,
            img_height=img_h,
            face_padding=face_padding,
            face_upper_shift=face_upper_shift,
        )
        face_pil = _crop_and_resize(frame_pil, face_box, face_crop_size, face_crop_size)
        face_frames.append(face_pil)

    target_aspect = height / width
    ref_img = Image.open(reference_image_path).convert("RGB")
    ref_crop = _center_crop_box(ref_img.size[0], ref_img.size[1], target_aspect)
    input_image = _crop_and_resize(ref_img, ref_crop, height, width)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "crop_box.json").write_text(
            json.dumps(
                {
                    "motion_video_path": str(motion_video_path),
                    "reference_image_path": str(reference_image_path),
                    "original_motion_size": {"width": int(img_w), "height": int(img_h)},
                    "target_size": {"width": int(width), "height": int(height)},
                    "num_frames": int(num_frames),
                    "animate_len": int(animate_len),
                    "crop_box": {"x1": int(crop_box[0]), "y1": int(crop_box[1]), "x2": int(crop_box[2]), "y2": int(crop_box[3])},
                },
                indent=2,
            )
        )
        Image.fromarray(frames_rgb[0]).convert("RGB").save(debug_dir / "motion_0.png")
        _crop_and_resize(Image.fromarray(frames_rgb[0]).convert("RGB"), crop_box, height, width).save(debug_dir / "motion_cropped_0.png")
        input_image.save(debug_dir / "input_image.png")
        pose_frames[0].save(debug_dir / "pose_0.png")
        face_frames[0].save(debug_dir / "face_0.png")

    return WanAnimateInputs(
        input_image=input_image,
        animate_pose_video=pose_frames[:animate_len],
        animate_face_video=face_frames[:animate_len],
        num_frames=num_frames,
        crop_box=crop_box,
    )
