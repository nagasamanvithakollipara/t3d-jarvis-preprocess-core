# core/utils/erp_box_utils.py

import math
from typing import List, Optional, Tuple, Dict, Union

import cv2
import numpy as np
import py360convert as p360
from sklearn.cluster import DBSCAN, AgglomerativeClustering


def spherical_iou(
    b1: np.ndarray,
    b2: np.ndarray,
    erp_w: int,
    erp_h: int,
) -> float:
    """
    Compute the Intersection‐over‐Union of two axis‐aligned boxes on an
    equirectangular projection, but measured properly on the sphere.

    Args:
        b1, b2: arrays [x1, y1, x2, y2] in pixel coords (ERP image).
        erp_w:  full panorama width in pixels.
        erp_h:  full panorama height in pixels.

    Returns:
        sIoU: float in [0,1].
    """

    # Helper: convert pixel to longitude/latitude in radians
    def px_to_lonlat(x: float, y: float) -> tuple[float, float]:
        # longitude: [−π, +π]
        lon = (x / erp_w) * 2 * np.pi - np.pi
        # latitude: [−π/2, +π/2]
        lat = np.pi / 2 - (y / erp_h) * np.pi
        return lon, lat

    # Extract lon/lat corners for each box
    def rect_params(box: np.ndarray):
        x1, y1, x2, y2 = box
        # four corners (we only need min/max in lon and in sin(lat))
        lon1, lat1 = px_to_lonlat(x1, y1)
        lon2, lat2 = px_to_lonlat(x2, y2)
        # ensure lon_min < lon_max, allowing seam‐crossing
        dlon = lon2 - lon1
        if dlon < 0:
            dlon += 2 * np.pi
        lon_min = lon1
        lon_max = lon1 + dlon
        # sine of latitude gives correct area weighting
        s1, s2 = np.sin(lat1), np.sin(lat2)
        sin_min, sin_max = min(s1, s2), max(s1, s2)
        return lon_min, lon_max, sin_min, sin_max

    # Get spherical‐rectangle params
    l1_min, l1_max, s1_min, s1_max = rect_params(b1)
    l2_min, l2_max, s2_min, s2_max = rect_params(b2)

    # Compute overlap in lon (radians) and in sin(lat)
    lon_overlap = max(0.0, min(l1_max, l2_max) - max(l1_min, l2_min))
    sin_overlap = max(0.0, min(s1_max, s2_max) - max(s1_min, s2_min))
    if lon_overlap <= 0 or sin_overlap <= 0:
        return 0.0

    # Spherical patch area on unit sphere = Δlon * Δ(sin lat)
    area_int = lon_overlap * sin_overlap
    area1 = (l1_max - l1_min) * (s1_max - s1_min)
    area2 = (l2_max - l2_min) * (s2_max - s2_min)

    return float(area_int / (area1 + area2 - area_int))


def hard_spherical_nms(
    dets: List[Dict],
    iou_thresh: float,
    erp_w: int,
    erp_h: int,
) -> List[Dict]:
    """
    Perform hard non‐maximum suppression using a spherical IoU metric.

    Args:
        dets:       List of detections, each a dict with
                      - "box": np.ndarray([x1,y1,x2,y2], float)
                      - "score": float
                      - any other keys
        iou_thresh: IoU threshold for suppression
        erp_w:      Width of the ERP image in pixels
        erp_h:      Height of the ERP image in pixels

    Returns:
        A filtered list of detections where no two boxes have spherical IoU ≥ iou_thresh.
    """
    keep: List[Dict] = []
    # sort by descending score
    remaining = sorted(dets, key=lambda d: d["score"], reverse=True)
    while remaining:
        top = remaining.pop(0)
        keep.append(top)
        filtered: List[Dict] = []
        for d in remaining:
            iou = spherical_iou(top["box"], d["box"], erp_w, erp_h)
            if iou < iou_thresh:
                filtered.append(d)
        remaining = filtered
    return keep


def extract_cube_faces(erp_bgr: np.ndarray, crop_size: int) -> List[np.ndarray]:
    """
    Split an ERP BGR image into 6 cubemap faces (BGR), in order:
        [Front(0°), Right(90°), Back(180°), Left(-90°), Up(90°), Down(-90°)].
        ┌────────┬────────┬────────┐
        │ Front  │ Right  │ Back   │
        ├────────┼────────┼────────┤
        │ Left   │ Up     │ Down   │
        └────────┴────────┴────────┘
    """
    erp_rgb = cv2.cvtColor(erp_bgr, cv2.COLOR_BGR2RGB)
    faces_rgb = p360.e2c(erp_rgb, crop_size, mode="bilinear", cube_format="list")
    return [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in faces_rgb]


def reproject_cube_box(
    face_idx: int,
    box_norm: Tuple[float, float, float, float],
    crop_size: int,
    erp_w: int,
    erp_h: int,
) -> Optional[np.ndarray]:
    """
    Given a normalized box [cx, cy, w, h] on one cube face, reproject it back
    into the ERP panorama and return the pixel bbox [x1, y1, x2, y2],
    or None if the box does not map to any ERP pixels.
    """
    # Unpack normalized center+size
    cx, cy, w_norm, h_norm = box_norm

    # Convert to pixel coordinates on the face
    center_x = cx * crop_size
    center_y = cy * crop_size
    half_w = (w_norm * crop_size) / 2
    half_h = (h_norm * crop_size) / 2

    # Clamp to face bounds
    x_min = int(np.clip(center_x - half_w, 0, crop_size - 1))
    x_max = int(np.clip(center_x + half_w, 0, crop_size - 1))
    y_min = int(np.clip(center_y - half_h, 0, crop_size - 1))
    y_max = int(np.clip(center_y + half_h, 0, crop_size - 1))

    # Build six face‐masks, with only this face non‐zero
    face_masks = [np.zeros((crop_size, crop_size), np.uint8) for _ in range(6)]
    face_masks[face_idx][y_min:y_max, x_min:x_max] = 255

    # Warp the mask(s) back to ERP
    erp_mask = p360.c2e(
        face_masks,
        erp_h,
        erp_w,
        mode="nearest",
        cube_format="list",
    )

    # Find all nonzero ERP pixels
    coords = np.argwhere(erp_mask > 0)
    if coords.size == 0:
        return None

    ys, xs = coords[:, 0], coords[:, 1]
    # Return [x_min, y_min, x_max, y_max] over those pixels
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=float)


def extract_nfov_crops(
    erp_bgr: np.ndarray,
    fov_deg: Union[float, List[float]],
    stride_yaw: float,
    out_hw: Tuple[int, int],
    pitch_angles: List[float],
    yaw_offsets: Optional[List[float]] = None,
    include_seam: bool = True,
) -> List[Tuple[float, float, float, np.ndarray]]:
    """
    Generate NFoV tiles from an ERP image.

    Sweep yaw ∈ [–180, +180) in steps of `stride_yaw`,
    sample ONLY the explicit `pitch_angles` (deg),
    optionally apply per-pitch yaw offsets,
    and optionally include the +180° “seam” yaw.

    Args:
        erp_bgr:        H×W×3 BGR ERP image.
        fov_deg:        horizontal & vertical FOV in degrees (square crop),
                        OR list of FOVs, one per pitch angle.
        stride_yaw:     step (deg) between successive yaw centers.
        out_hw:         (height, width) of each NFoV crop.
        pitch_angles:   exact list of pitch centers (deg) to sample.
        yaw_offsets:    optional list of yaw offsets (deg) per pitch angle
                        (defaults to 0° for every ring).
        include_seam:   whether to also sample yaw = +180° seam view.

    Returns:
        List of (yaw_deg, pitch_deg, fov_deg, crop_bgr) tuples.
    """
    H, W = erp_bgr.shape[:2]
    erp_rgb = cv2.cvtColor(erp_bgr, cv2.COLOR_BGR2RGB)

    # build per-pitch FOV list
    if isinstance(fov_deg, (list, tuple)):
        fovs = list(fov_deg)
    else:
        fovs = [fov_deg] * len(pitch_angles)

    # build per-pitch yaw-offset list
    if yaw_offsets is None:
        y_offsets = [0.0] * len(pitch_angles)
    else:
        y_offsets = yaw_offsets

    crops: List[Tuple[float, float, float, np.ndarray]] = []
    # base yaw grid (no offset)
    base_yaws = np.arange(-180.0, 180.0, stride_yaw)

    for idx, pitch in enumerate(pitch_angles):
        fov = fovs[idx]
        offset = y_offsets[idx]

        # clamp pitch so that the NFoV stays on the sphere
        pitch_clamped = float(max(min(pitch, 90.0 - fov / 2.0), -90.0 + fov / 2.0))

        # apply yaw offset and normalize into [-180, +180)
        yaws = []
        for y in base_yaws:
            y_o = y + offset
            y_norm = ((y_o + 180.0) % 360.0) - 180.0
            yaws.append(y_norm)

        # optionally add the seam yaw (offset + 180°)
        if include_seam:
            seam = ((180.0 + offset + 180.0) % 360.0) - 180.0
            if seam not in yaws:
                yaws.append(seam)

        # generate NFoV crop for each yaw
        for yaw in yaws:
            crop_rgb = p360.e2p(
                erp_rgb,
                fov_deg=(fov, fov),
                u_deg=yaw,
                v_deg=pitch_clamped,
                out_hw=out_hw,
                mode="bilinear",
            )
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            crops.append((yaw, pitch_clamped, fov, crop_bgr))

    return crops


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def project_nfov_box(
    box_norm: Tuple[float, float, float, float],
    yaw: float,
    pitch: float,
    crop_size: int,
    erp_w: int,
    erp_h: int,
    fov_deg: Union[float, List[float]],
) -> np.ndarray:
    """
    Map a normalized box [cx,cy,w,h] in an NFoV tile back to ERP pixel bbox.

    Args:
        box_norm:  (cx,cy,w,h) in normalized NFoV coords.
        yaw:       center yaw of the NFoV tile (deg).
        pitch:     center pitch of the NFoV tile (deg).
        crop_size: side‐length in pixels of the square NFoV tile.
        erp_w:     ERP panorama width in pixels.
        erp_h:     ERP panorama height in pixels.
        fov_deg:   horizontal & vertical FoVs of the NFoV tile in degrees,
                   or a list thereof (in which case the first value is used).

    Returns:
        [x_min, y_min, x_max, y_max] pixel coords on the ERP image.
    """
    # if a list of FoVs was passed, take the first element
    if isinstance(fov_deg, (list, tuple)):
        fov = float(fov_deg[0])
    else:
        fov = float(fov_deg)

    # unpack normalized center & half‐sizes
    cx, cy, w, h = box_norm
    px, py = cx * crop_size, cy * crop_size
    hw, hh = (w * crop_size / 2), (h * crop_size / 2)

    # compute the four corner pixel coords in NFoV
    corners = [(px + dx, py + dy) for dx in (-hw, hw) for dy in (-hh, hh)]

    # precompute rotations
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    tan_f = math.tan(math.radians(fov / 2))

    us, vs = [], []
    for corner_x, corner_y in corners:
        # map from pixel → camera plane coordinates (x right, y up)
        x_cam = (corner_x / crop_size - 0.5) * 2 * tan_f
        y_cam = -(corner_y / crop_size - 0.5) * 2 * tan_f

        # apply camera→world rotation
        R = _rot_y(yaw_rad) @ _rot_x(-pitch_rad)
        vx, vy, vz = R @ np.array([x_cam, y_cam, 1.0], dtype=float)

        # convert direction vector → spherical lon/lat
        lon = math.atan2(vx, vz)
        lat = math.asin(vy / np.linalg.norm([vx, vy, vz]))

        # map lon/lat → ERP pixel coords
        u = (lon + math.pi) / (2 * math.pi) * erp_w
        v = (math.pi / 2 - lat) / math.pi * erp_h

        us.append(u)
        vs.append(v)

    # return axis‐aligned bbox in ERP coords
    return np.array([min(us), min(vs), max(us), max(vs)], dtype=float)


def box_to_unit_vector(box: List[float], W: int, H: int) -> np.ndarray:
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2

    lon = (x_center / W) * 2 * np.pi - np.pi  # yaw ∈ [-π, π]
    lat = np.pi / 2 - (y_center / H) * np.pi  # pitch ∈ [-π/2, π/2]

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return np.array([x, y, z])


def spherical_clustering(
    dets: List[Dict],
    W: int,
    H: int,
    eps_degrees: float,
    min_samples=2,
) -> List[Dict]:
    if not dets:
        return []

    # Convert to unit vectors
    vectors = np.array([box_to_unit_vector(d["box"], W, H) for d in dets])

    # Compute angular distance as eps for DBSCAN
    eps = np.deg2rad(eps_degrees)

    # Use cosine similarity as distance metric
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(vectors)
    labels = clustering.labels_

    # Group detections by cluster label
    clustered = {}
    for i, label in enumerate(labels):
        if label not in clustered:
            clustered[label] = []
        clustered[label].append(dets[i])

    # For each cluster, keep the highest score box
    merged_dets = []
    for cluster_dets in clustered.values():
        best = max(cluster_dets, key=lambda x: x["score"])
        merged_dets.append(best)

    return merged_dets


def agglomerative_spherical_nms(
    dets: List[Dict], W: int, H: int, iou_threshold: float = 0.3
) -> List[Dict]:
    """
    Cluster detections by spherical IoU ≥ iou_threshold using
    Agglomerative Clustering on the precomputed distance matrix,
    then keep the highest‐score box per cluster.
    """
    N = len(dets)
    if N == 0:
        return []

    # Build distance matrix: D[i,j] = 1 - IoU(i,j)
    dist = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            # use your existing spherical_iou implementation
            b1 = np.asarray(dets[i]["box"], dtype=float)
            b2 = np.asarray(dets[j]["box"], dtype=float)
            iou = spherical_iou(b1, b2, erp_w=W, erp_h=H)
            d = 1.0 - iou
            dist[i, j] = d
            dist[j, i] = d

    # Agglomerative clustering on precomputed distances
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - iou_threshold,
    )
    labels = clustering.fit_predict(dist)

    # Group detections by cluster label
    clusters: Dict[int, List[Dict]] = {}
    for det, lbl in zip(dets, labels):
        clusters.setdefault(lbl, []).append(det)

    # For each cluster, keep the box with the highest score
    merged: List[Dict] = []
    for cluster in clusters.values():
        best = max(cluster, key=lambda d: d["score"])
        merged.append(best)

    return merged
