import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image

from common.logger import logger
from core.config import PipelineConfig
from core.vlm.llm_wrapper import LLMClientWrapper
from core.utils.erp_box_utils import (
    extract_cube_faces,
    extract_nfov_crops,
)
from core.utils.pipeline_utils import strip_json_fences


class VLMObjectCounter:
    """
    VLM-based object counter over ERP video.  Supports:
      - 'full' : count per full frame
      - 'cube' : count per cubemap faces then aggregate
      - 'nfov' : count per NFoV views then aggregate

    Outputs a JSON file with one record per sampled frame:
      {"frame_id": int, "counts": {label: count, ...}}
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.llm = LLMClientWrapper(cfg)
        self.tpl = cfg.templates.count_objects

    def _dispatch(
        self,
        prompts: List[str],
        imgs: List[Image.Image],
        label: str,
    ) -> List[str]:
        """
        Send prompts/images to LLM either via OpenAI batch or parallel.
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
        Sample frames, count objects under each mode, write detections.json.
        """
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
        out_dir = Path(output_dir) / f"{stem}_{mode}_counts"
        out_dir.mkdir(parents=True, exist_ok=True)

        flat_prompts: List[str] = []
        flat_imgs: List[Image.Image] = []
        frame_map: List[Any] = []  # int or (int,face_idx) or (int,yaw,pitch)

        # build tasks per mode
        if mode == "full":
            for fid, frame in tasks:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                flat_prompts.append(self.tpl.format(instruction=instruction))
                flat_imgs.append(pil)
                frame_map.append(fid)

        elif mode == "cube":
            for fid, frame in tasks:
                faces = extract_cube_faces(frame, self.cfg.face_size)
                for i, face in enumerate(faces):
                    if i not in self.cfg.cube_faces:
                        continue
                    pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    flat_prompts.append(self.tpl.format(instruction=instruction))
                    flat_imgs.append(pil)
                    frame_map.append((fid, i))

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
                    flat_prompts.append(self.tpl.format(instruction=instruction))
                    flat_imgs.append(pil)
                    frame_map.append((fid, yaw, pitch, fov))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        label = {
            "full": "full-frame",
            "cube": "cube-face",
            "nfov": "nfov-view",
        }[mode]
        raw_list = self._dispatch(flat_prompts, flat_imgs, label)

        # aggregate counts per frame
        per_frame_counts: Dict[int, Dict[str, int]] = {fid: {} for fid, _ in tasks}
        for key, raw in zip(frame_map, raw_list):

            cleaned = strip_json_fences(raw)
            try:
                data = json.loads(cleaned)
                if not isinstance(data, dict):
                    raise ValueError("expected top-level JSON object")

                from pprint import pprint

                pprint(data, indent=2, width=80)

            except Exception as e:
                logger.warning(
                    f"VLM JSON parse failed for task {key}: {e}\n"
                    f"--- raw response ---\n{raw}\n"
                    f"--- cleaned text  ---\n{cleaned}"
                )
                continue

            if mode == "full":
                fid = key
            else:
                fid = key[0]

            # merge into per_frame_counts[fid]
            frame_counts = per_frame_counts[fid]
            for lbl, cnt in data.items():
                frame_counts[lbl] = frame_counts.get(lbl, 0) + int(cnt)

        # write out JSON records
        records = []
        for fid, _ in tasks:
            records.append({"frame_id": fid + 1, "counts": per_frame_counts[fid]})
        out_file = out_dir / "counts.json"
        with open(out_file, "w") as f:
            json.dump(records, f, indent=2)
        logger.info(f"Counts written to {out_file}")
