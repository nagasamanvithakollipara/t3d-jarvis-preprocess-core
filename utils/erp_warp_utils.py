# core/utils/erp_warp_utils.py

import torch
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional, Union, Sequence, Literal

import py360convert as p360
from common.logger import logger


FACE_NAMES = ["front", "right", "back", "left", "up", "down"]


def annotate(
    image_source: np.ndarray,
    boxes: Union[torch.Tensor, Sequence[Sequence[float]]],
    logits: Union[torch.Tensor, Sequence[float]],
    phrases: List[str],
) -> np.ndarray:
    """
    Draw bounding boxes and phrases on an RGB image and return as BGR for OpenCV.
    Accepts boxes/logits as either Python lists or Tensors.
    """
    # Convert tensors → lists
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy().tolist()
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy().tolist()

    img = image_source.copy()
    for (x0, y0, x1, y1), score, phrase in zip(boxes, logits, phrases):
        # Comment out green bounding box drawing to remove green boxes
        # cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        # label = f"{phrase}:{score:.2f}"
        # cv2.putText(
        #     img,
        #     label,
        #     (int(float(x0)), int(max(float(y0) - 10, 0))),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 255, 0),
        #     1,
        # )
        pass  # Skip drawing boxes and labels
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def erp_to_cubemap(frame: Image.Image, face_size: int) -> Dict[str, Image.Image]:
    """
    Convert an equirectangular (ERP) panorama to a cube‐map of six faces.

    Args:
        frame:      A PIL Image in equirectangular projection (width ≈ 2×height).
        face_size:  Output size (pixels) for each square face.
        mode:       Resampling mode, e.g. "bilinear" or "nearest".

    Returns:
        A dict mapping each face name → PIL Image (face_size×face_size):
          {
            "front": Image,  # yaw=0°
            "right": Image,  # yaw=+90°
            "back":  Image,  # yaw=180°
            "left":  Image,  # yaw=-90°
            "up":    Image,  # pitch=+90°
            "down":  Image   # pitch=-90°
          }

    Raises:
        ValueError:   If face_size ≤ 0.
        RuntimeError: If the projection fails.
    """
    w, h = frame.size
    if face_size <= 0:
        raise ValueError(f"erp_to_cubemap: face_size must be > 0 (got {face_size})")
    # Optionally warn if not true 2:1 ERP
    if abs(w / h - 2.0) > 1e-3:
        logger.warning(f"erp_to_cubemap: input not 2:1 panorama ({w}×{h})")

    # Convert to NumPy array
    pano = np.asarray(frame)
    try:
        cube_faces = p360.e2c(pano, face_size, mode="bilinear", cube_format="list")
    except Exception as e:
        raise RuntimeError(f"erp_to_cubemap: cubemap projection failed: {e}") from e

    # Map faces in standard order
    face_names = ["front", "right", "back", "left", "up", "down"]
    faces: Dict[str, Any] = {}
    for idx, name in enumerate(face_names):
        arr = cube_faces[idx]
        # ensure dtype=uint8
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        faces[name] = Image.fromarray(arr)

    return faces


def erp_to_nfov(
    frame: Image.Image,
    fov_x: float = 90.0,
    stride_yaw: float = 90.0,
    stride_pitch: float = 90.0,
    out_hw: Optional[Tuple[int, int]] = None,
) -> List[Tuple[float, float, Image.Image]]:
    """
    Slide a perspective “camera” across an ERP panorama.

    Args:
        frame:        PIL.Image in equirectangular (2:1) projection.
        fov_x:        Horizontal field-of-view in degrees.
        stride_yaw:   Step between yaw angles (0–360°).
        stride_pitch: Step between pitch angles (−90°–+90°).
        out_hw:       (height, width) of each NFOV crop in pixels.
                      If None, defaults to (face_size, face_size)
                      where face_size = frame.width // 4.

    Returns:
        List of (yaw, pitch, crop_image), where each crop_image is
        a PIL.Image of size out_hw.
    """
    pano = np.asarray(frame)
    H, W = frame.height, frame.width

    # compute vertical FOV to preserve aspect ratio
    fov_y = fov_x * (H / W)
    fov_deg = (fov_x, fov_y)

    if out_hw is None:
        face = max(1, W // 4)
        out_hw = (face, face)

    # valid pitch range so we don’t go over the poles
    min_pitch = -90.0 + fov_y / 2.0
    max_pitch = 90.0 - fov_y / 2.0

    yaw_angles = np.arange(0.0, 360.0, stride_yaw, dtype=float)
    pitch_angles = np.arange(max_pitch, min_pitch - 1e-6, -stride_pitch, dtype=float)

    crops: List[Tuple[float, float, Image.Image]] = []
    for pitch in pitch_angles:
        for yaw in yaw_angles:
            try:
                persp = p360.e2p(
                    pano,
                    fov_deg,
                    yaw,
                    pitch,
                    out_hw,
                    0.0,
                    "bilinear",
                )
            except Exception as e:
                raise RuntimeError(f"erp_to_nfov projection failed: {e}") from e

            if persp.dtype != np.uint8:
                persp = np.clip(persp, 0, 255).astype(np.uint8)
            crops.append((yaw, pitch, Image.fromarray(persp)))

    return crops


def cubemap_to_erp(
    faces: Dict[str, Image.Image],
    face_size: int,
    erp_h: int,
    erp_w: int,
    mode: Literal["bilinear", "nearest"] = "nearest",
    cube_format: Literal["list", "dict", "horizon", "dice"] = "list",
) -> Image.Image:
    """
    Reproject six cube faces back into an equirectangular (ERP) panorama.

    Args:
        faces:     Mapping from face name to PIL.Image (square faces of size face_size).
                   Expected keys: ['front', 'right', 'back', 'left', 'up', 'down'].
        face_size: Pixel dimension of each square face (must be > 0).
        erp_h:     Output ERP height in pixels (must be > 0).
        erp_w:     Output ERP width in pixels (must be > 0).
        mode:      Interpolation mode: "bilinear" or "nearest".
        cube_format: Format of the input cubemap:
                     - "list":  faces as a list in standard order
                     - "dict":  faces as a dict {name: array}
                     - "horizon": specialized format
                     - "dice":   4×3 grid arrangement

    Returns:
        A PIL.Image of shape (erp_w × erp_h), covering 360°×180° at the requested resolution.

    Raises:
        ValueError:   If face_size ≤ 0, erp_h ≤ 0, erp_w ≤ 0, or required faces missing.
        RuntimeError: If reprojection fails.
    """
    # Validate inputs
    if face_size <= 0:
        raise ValueError(f"cubemap_to_erp: face_size must be > 0 (got {face_size})")
    if erp_h <= 0 or erp_w <= 0:
        raise ValueError(
            f"cubemap_to_erp: erp_h/erp_w must be > 0 (got {erp_h}×{erp_w})"
        )

    missing = [f for f in FACE_NAMES if f not in faces]
    if missing:
        raise ValueError(f"cubemap_to_erp: missing faces: {missing}")

    try:
        if cube_format == "dict":
            cube_arrs = {name: np.asarray(faces[name]) for name in FACE_NAMES}
            pano_np = p360.c2e(
                cube_arrs,
                erp_h,
                erp_w,
                mode=mode,
                cube_format="dict",
            )
        else:
            cube_arrs = [np.asarray(faces[name]) for name in FACE_NAMES]
            pano_np = p360.c2e(
                cube_arrs,
                erp_h,
                erp_w,
                mode=mode,
                cube_format="list",
            )
    except Exception as e:
        logger.error(f"cubemap_to_erp: reprojection failed: {e}")
        raise RuntimeError(f"cubemap_to_erp: reprojection failed: {e}") from e

    try:
        pano_uint8 = np.clip(pano_np, 0, 255).astype(np.uint8)
        return Image.fromarray(pano_uint8)
    except Exception as e:
        logger.error(f"cubemap_to_erp: failed to convert to image: {e}")
        raise


def filter_face_detections(
    boxes: Any,
    scores: Any,
    phrases: List[str],
    face_w: int,
    face_h: int,
    max_face_frac: float = 0.9,
    min_score: float = 0.0,
) -> Tuple[List[List[float]], np.ndarray, List[str]]:
    """
    Filter out detections that either
     - Have confidence < min_score, or
     - Cover more than max_face_frac of the face in both width & height.

    Args:
      boxes:       Tensor[N,4] or List of [x0, y0, x1, y1]
      scores:      Tensor[N] or List of floats
      phrases:     List[str] of length N
      face_w:      Width of the face image in pixels
      face_h:      Height of the face image in pixels
      max_face_frac: Reject boxes whose width AND height exceed this fraction of face
      min_score:   Reject boxes with confidence < this

    Returns:
      (filtered_boxes, filtered_scores, filtered_phrases)
    """
    # to numpy
    if hasattr(boxes, "cpu"):
        boxes_np = boxes.cpu().numpy()
    else:
        boxes_np = np.array(boxes, dtype=float)
    if hasattr(scores, "cpu"):
        scores_np = scores.cpu().numpy()
    else:
        scores_np = np.array(scores, dtype=float)

    keep_boxes, keep_scores, keep_phrases = [], [], []
    for (x0, y0, x1, y1), score, phrase in zip(boxes_np, scores_np, phrases):
        w = x1 - x0
        h = y1 - y0
        if score < min_score:
            continue
        # if it covers almost the whole face, drop it
        if w > max_face_frac * face_w and h > max_face_frac * face_h:
            continue
        keep_boxes.append([float(x0), float(y0), float(x1), float(y1)])
        keep_scores.append(float(score))
        keep_phrases.append(phrase)

    return keep_boxes, np.array(keep_scores), keep_phrases
