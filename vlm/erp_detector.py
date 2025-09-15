import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image

from common.logger import logger
from core.config import PipelineConfig

from core.utils.erp_box_utils import (
    extract_cube_faces,
    reproject_cube_box,
    extract_nfov_crops,
    project_nfov_box,
)
from core.utils.pipeline_utils import (
    strip_json_fences,
    safe_parse_json,
)
from core.utils.detection_utils import hard_nms, filter_seam_artifacts
from .llm_wrapper import LLMClientWrapper


class VLMDetector:
    """
    VLM-based object detector over ERP video.  Supports:
      - 'full'  : batch or parallel full-frame calls
      - 'cube'  : batch/parallel per-face cubemap
      - 'nfov'  : batch/parallel per-view NFoV

    Writes per-sampled-frame:
      - annotated JPEG under out_dir/<video_stem>_<mode>/
      - collects one JSON record per frame with {frame_id, count, bboxes}.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.llm = LLMClientWrapper(cfg)
        # prompt template
        self.vlm_tpl = cfg.templates.vlm_detection

    def _denormalize(self, nb: List[float], w: int, h: int) -> List[float]:
        """
        Convert normalized [x0,y0,x1,y1] → pixel coords [x0*w, y0*h, x1*w, y1*h].
        """
        x0, y0, x1, y1 = nb
        if all(0.0 <= v <= 1.0 for v in nb):
            return [x0 * w, y0 * h, x1 * w, y1 * h]
        else:
            return [x0, y0, x1, y1]

    def _dispatch(
        self,
        prompts: List[str],
        imgs: List[Image.Image],
        label: str,
    ) -> List[str]:
        """
        Send a batch of prompt/image pairs to the LLM:

          - If using OpenAI *and* openai_batch_mode → use generate_openai_batch
          - Otherwise → use generate_parallel with configured concurrency.

        Args:
            prompts: list of text prompts.
            imgs:    corresponding list of PIL images.
            label:   short tag for logging ("full-frame", "cube-face", "nfov-view").

        Returns:
            List of raw response strings (JSON blobs).
        """
        llm_type = self.cfg.llm.lower()
        use_openai = llm_type == "openai"
        logger.info(f"Using '{self.cfg.llm}' LLM")

        # Batch API path for OpenAI
        if use_openai and getattr(self.cfg, "openai_batch_mode", False):
            n = len(prompts)
            logger.info(f"OpenAI batch over {n} {label} tasks")
            return self.llm.generate_openai_batch(prompts, imgs)

        # Determine concurrency based on LLM type
        if use_openai:
            workers = self.cfg.openai_concurrency
        elif llm_type == "ollama":
            workers = self.cfg.ollama_concurrency
        elif llm_type == "qwen":
            workers = self.cfg.qwen_concurrency
        else:
            # Fallback to single‐threaded
            workers = 1

        n = len(prompts)
        logger.info(
            f"Running parallel sync on {label} with {workers} workers over {n} tasks"
        )
        return self.llm.generate_parallel(prompts, imgs, max_workers=workers)

    def run_video(
        self,
        video_path: str,
        output_dir: str,
        instruction: str,
    ) -> None:
        """
        Main entrypoint: sample frames, run VLM detection in the selected mode,
        reproject & merge bboxes, annotate frames, and dump detections.json.
        """
        # 1) Sample frames at cfg.frame_rate
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(fps / self.cfg.frame_rate))

        tasks: List[Tuple[int, np.ndarray]] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                tasks.append((idx // step, frame.copy()))
            idx += 1
        cap.release()

        mode = self.cfg.erp_mode.lower()
        stem = Path(video_path).stem
        out_dir = Path(output_dir) / f"{stem}_{mode}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2) Build flattened lists of prompts/images and a map back to frame_id
        flat_prompts: List[str] = []
        flat_imgs: List[Image.Image] = []
        frame_map: List[Any] = []  # either int or tuple

        if mode == "full":
            for fid, frame in tasks:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                flat_prompts.append(self.vlm_tpl.format(instruction=instruction))
                flat_imgs.append(pil)
                frame_map.append(fid)

        elif mode == "cube":
            for fid, frame in tasks:
                faces = extract_cube_faces(frame, self.cfg.face_size)
                for face_idx, face_bgr in enumerate(faces):
                    if face_idx not in self.cfg.cube_faces:
                        continue
                    pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
                    flat_prompts.append(self.vlm_tpl.format(instruction=instruction))
                    flat_imgs.append(pil)
                    frame_map.append((fid, face_idx))

        elif mode == "nfov":
            for fid, frame in tasks:
                crops = extract_nfov_crops(
                    frame,
                    self.cfg.nfov_fov_deg,
                    self.cfg.nfov_stride_yaw,
                    self.cfg.nfov_out_hw,
                    pitch_angles=self.cfg.nfov_pitch_angles,
                    yaw_offsets=self.cfg.nfov_yaw_offset,
                    include_seam=self.cfg.nfov_include_seam,
                )
                for yaw, pitch, fov, crop_bgr in crops:
                    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                    flat_prompts.append(self.vlm_tpl.format(instruction=instruction))
                    flat_imgs.append(pil)
                    frame_map.append((fid, yaw, pitch, fov))

        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

        # 3) Fire off LLM calls in batch or parallel
        label = {
            "full": "full-frame",
            "cube": "cube-face",
            "nfov": "nfov-view",
        }[mode]
        raw_list = self._dispatch(flat_prompts, flat_imgs, label)

        # 4) Parse JSON & reproject to ERP pixel coords
        per_frame: Dict[int, List[Dict]] = {fid: [] for fid, _ in tasks}
        for key, raw in zip(frame_map, raw_list):
            cleaned = strip_json_fences(raw)
            try:
                objs = safe_parse_json(cleaned)
                if not isinstance(objs, list):
                    raise ValueError("expected top‐level JSON list")

                from pprint import pprint

                pprint(raw, indent=2, width=80)
            except Exception as e:
                logger.warning(
                    f"VLM JSON parse failed for task {key}: {e}\n"
                    f"--- raw response ---\n{raw}\n"
                    f"--- cleaned text  ---\n{cleaned}"
                )
                continue

            if mode == "full":
                fid = key
                H, W = tasks[fid][1].shape[:2]
                for obj in objs:
                    nb = obj.get("bbox", [])
                    if len(nb) == 4:
                        per_frame[fid].append(
                            {
                                "label": obj.get("label", ""),
                                "box": self._denormalize(nb, W, H),
                            }
                        )

            elif mode == "cube":
                fid, face_idx = key
                H, W = tasks[fid][1].shape[:2]
                for obj in objs:
                    nb = obj.get("bbox", [])
                    if len(nb) == 4:
                        # convert to normalized [cx,cy,w,h]
                        ph = pw = self.cfg.face_size
                        cx = ((nb[0] + nb[2]) / 2) / pw
                        cy = ((nb[1] + nb[3]) / 2) / ph
                        wn = (nb[2] - nb[0]) / pw
                        hn = (nb[3] - nb[1]) / ph
                        erp_box = reproject_cube_box(
                            face_idx, (cx, cy, wn, hn), pw, W, H
                        )
                        if erp_box is not None:
                            per_frame[fid].append(
                                {
                                    "label": obj.get("label", ""),
                                    "box": erp_box.tolist(),
                                }
                            )

            else:  # nfov
                fid, yaw, pitch, fov = key
                H, W = tasks[fid][1].shape[:2]
                crop_size = self.cfg.nfov_out_hw[1]
                for obj in objs:
                    nb = obj.get("bbox", [])
                    if len(nb) == 4:
                        cx = ((nb[0] + nb[2]) / 2) / crop_size
                        cy = ((nb[1] + nb[3]) / 2) / crop_size
                        wn = (nb[2] - nb[0]) / crop_size
                        hn = (nb[3] - nb[1]) / crop_size
                        erp_box = project_nfov_box(
                            (cx, cy, wn, hn),
                            yaw=yaw,
                            pitch=pitch,
                            crop_size=crop_size,
                            erp_w=W,
                            erp_h=H,
                            fov_deg=fov,
                        )
                        per_frame[fid].append(
                            {
                                "label": obj.get("label", ""),
                                "box": erp_box.tolist(),
                            }
                        )

        # 5) Per-frame NMS, seam-filter, annotate & record

        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        json_records: List[Dict] = []
        for fid, frame in tasks:
            frame_num = fid + 1
            recs = per_frame[fid]
            dets = [{"box": r["box"], "score": 1.0, "phrase": r["label"]} for r in recs]
            dets = hard_nms(dets, self.cfg.nms_iou_thresh)
            dets = filter_seam_artifacts(
                dets,
                fw=frame.shape[1],
                fh=frame.shape[0],
                max_w_frac=self.cfg.seam_max_w_frac,
                max_h_frac=self.cfg.seam_max_h_frac,
            )

            # build per-label lists, annotate & save
            label_boxes: Dict[str, List[List[float]]] = {}
            ann = frame.copy()
            for d in dets:
                label = d["phrase"]
                box = [float(x) for x in d["box"]]
                label_boxes.setdefault(label, []).append(box)

                # draw it
                x0, y0, x1, y1 = box
                cv2.rectangle(
                    ann,
                    (int(x0), int(y0)),
                    (int(x1), int(y1)),
                    (0, 255, 0),
                    2,
                )

            # save annotated frame
            out_path = frames_dir / f"frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(out_path), ann)

            # build counts per label
            counts = {label: len(bxs) for label, bxs in label_boxes.items()}

            json_records.append(
                {
                    "frame_id": frame_num,
                    "counts": counts,
                    "bboxes": label_boxes,
                }
            )

        # 6) Dump all detections
        with open(out_dir / "detections.json", "w") as f:
            json.dump(json_records, f, indent=2)
        logger.info(f"VLM detections & annotations written to {out_dir}")
