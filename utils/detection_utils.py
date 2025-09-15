# core/utils/detection_utils.py

from typing import List, Dict
import numpy as np


def compute_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def hard_nms(dets: List[Dict], iou_thresh: float) -> List[Dict]:
    ds = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep = []
    while ds:
        top = ds.pop(0)
        keep.append(top)
        ds = [d for d in ds if compute_iou(top["box"], d["box"]) < iou_thresh]
    return keep


def filter_seam_artifacts(
    dets: List[Dict], fw: int, fh: int, max_w_frac: float, max_h_frac: float
) -> List[Dict]:
    out = []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        w, h = x2 - x1, y2 - y1
        if (w / fw > max_w_frac) and (h / fh < max_h_frac):
            continue
        out.append(d)
    return out


def cluster_detections(dets: List[Dict], iou_thresh: float) -> List[Dict]:
    """
    Greedy IoU‐based clustering: group any detections whose IoU ≥ iou_thresh,
    then merge each group into a single box (union) with the max score.
    """
    if not dets:
        return []

    # sort by score descending
    dets_sorted = sorted(dets, key=lambda d: d["score"], reverse=True)
    used = [False] * len(dets_sorted)
    clusters = []

    for i, di in enumerate(dets_sorted):
        if used[i]:
            continue
        # start a new cluster
        cluster = [di]
        used[i] = True
        for j in range(i + 1, len(dets_sorted)):
            if used[j]:
                continue
            dj = dets_sorted[j]
            if compute_iou(di["box"], dj["box"]) >= iou_thresh:
                cluster.append(dj)
                used[j] = True
        clusters.append(cluster)

    # merge each cluster
    merged: List[Dict] = []
    for cluster in clusters:
        boxes = np.stack([c["box"] for c in cluster], axis=0)
        x1 = boxes[:, 0].min()
        y1 = boxes[:, 1].min()
        x2 = boxes[:, 2].max()
        y2 = boxes[:, 3].max()
        score = max(c["score"] for c in cluster)
        phrase = cluster[0]["phrase"]
        merged.append(
            {
                "box": np.array([x1, y1, x2, y2], dtype=float),
                "score": score,
                "phrase": phrase,
            }
        )

    return merged
