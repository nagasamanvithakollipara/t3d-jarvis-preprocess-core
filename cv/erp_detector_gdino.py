# core/utils/erp_detection_gdino.py
import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple, Union

from common.logger import logger
from core.config import PipelineConfig
from core.models.gdino_client import GroundingDINOEngine
from core.models.gdino_hf_client import GroundingDINOHFEngine
from core.models.sam_client import SAMClient
from core.utils.detection_utils import (
    # hard_nms,
    filter_seam_artifacts,
)
from core.utils.erp_box_utils import (
    extract_cube_faces,
    reproject_cube_box,
    extract_nfov_crops,
    project_nfov_box,
    hard_spherical_nms,
    agglomerative_spherical_nms,
)
from core.utils.erp_warp_utils import (
    erp_to_cubemap,
    cubemap_to_erp,
    annotate,
    filter_face_detections,
)
from core.utils.erp_warp_utils import FACE_NAMES
from core.utils.pipeline_utils import timeit


class ERPBoxDetector:
    """
    Bounding‐box detection + reprojection on ERP frames in 'full', 'cube', or 'nfov' mode.
    Supports batch inference via GroundingDINOEngine (local & HF).
    """

    def __init__(
        self,
        det_engine: Union[GroundingDINOEngine, GroundingDINOHFEngine],
        cfg: PipelineConfig,
        prompt: List[str],
        sam_client: Optional[SAMClient] = None,
    ):
        """
        Args:
            gd_engine:   Preconfigured GroundingDINOEngine.
            cfg:         PipelineConfig with mode, thresholds, face/nfov settings.
            prompt:      Text prompt for DINO.
            sam_client:  Optional SAMClient for mask refinement.
        """
        self.det_engine = det_engine
        self.cfg = cfg
        self.prompt = prompt
        self.sam_client = sam_client

    @timeit
    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Detect objects on an ERP frame, reproject their boxes back to ERP coords,
        apply NMS, seam filtering, and optional NFoV clustering.

        Args:
            frame_bgr: H×W×3 BGR input frame.

        Returns:
            List of dicts with keys:
              - "box":   np.ndarray([x1,y1,x2,y2], float)
              - "score": float
              - "phrase": str
        """
        H, W = frame_bgr.shape[:2]
        dets: List[Dict] = []
        mode = self.cfg.erp_mode.lower()

        # ─── FULL FRAME ─────────────────────────────────────────────
        if mode == "full":
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            raw_boxes, raw_scores, raw_phrases = self.det_engine.detect(
                pil,
                labels=self.prompt,
                box_threshold=self.cfg.gdino_box_threshold,
                text_threshold=self.cfg.gdino_text_threshold,
            )

            # filter & ensure tensor types
            boxes, scores, phrases = filter_face_detections(
                raw_boxes,
                raw_scores,
                raw_phrases,
                pil.width,
                pil.height,
                max_face_frac=0.95,  # Changed from 0.9 to allow larger objects
                min_score=self.cfg.gdino_box_threshold,
            )
            if not torch.is_tensor(boxes):
                boxes = torch.tensor(boxes, dtype=torch.float32)
            if not torch.is_tensor(scores):
                scores = torch.tensor(scores, dtype=torch.float32)

            # collect detections
            for (x0, y0, x1, y1), score, phrase in zip(
                boxes.cpu().numpy(),
                scores.cpu().numpy(),
                phrases,
            ):
                dets.append(
                    {
                        "box": np.array([int(x0), int(y0), int(x1), int(y1)], float),
                        "score": float(score),
                        "phrase": phrase,
                    }
                )

        # ─── CUBEMAP MODE ──────────────────────────────────────────
        elif mode == "cube":
            faces = extract_cube_faces(frame_bgr, self.cfg.face_size)
            # prepare PIL list
            face_idxs = self.cfg.cube_faces
            images = [
                Image.fromarray(cv2.cvtColor(faces[idx], cv2.COLOR_BGR2RGB))
                for idx in face_idxs
            ]

            # batch if multiple
            if len(images) > 1:
                logger.debug(
                    f"ERPBoxDetector: batch detecting {len(images)} cube faces"
                )
                boxes_list, scores_list, phrases_list = self.det_engine.detect_batch(
                    images,
                    labels=self.prompt,
                    box_threshold=self.cfg.gdino_box_threshold,
                    text_threshold=self.cfg.gdino_text_threshold,
                )
            else:
                logger.debug("ERPBoxDetector: single cube-face detection")
                raw = self.det_engine.detect(
                    images[0],
                    labels=self.prompt,
                    box_threshold=self.cfg.gdino_box_threshold,
                    text_threshold=self.cfg.gdino_text_threshold,
                )
                boxes_list = [raw[0]]
                scores_list = [raw[1]]
                phrases_list = [raw[2]]

            # process each face
            for idx, boxes, scores, phrases in zip(
                face_idxs, boxes_list, scores_list, phrases_list
            ):
                # filter & ensure tensor types
                boxes, scores, phrases = filter_face_detections(
                    boxes,
                    scores,
                    phrases,
                    self.cfg.face_size,
                    self.cfg.face_size,
                    max_face_frac=0.9,
                    min_score=self.cfg.gdino_box_threshold,
                )
                if not torch.is_tensor(boxes):
                    boxes = torch.tensor(boxes, dtype=torch.float32)
                if not torch.is_tensor(scores):
                    scores = torch.tensor(scores, dtype=torch.float32)

                # reproject each
                for (x0, y0, x1, y1), score, phrase in zip(
                    boxes.cpu().numpy(),
                    scores.cpu().numpy(),
                    phrases,
                ):
                    cx = (x0 + x1) / 2 / self.cfg.face_size
                    cy = (y0 + y1) / 2 / self.cfg.face_size
                    w_norm = (x1 - x0) / self.cfg.face_size
                    h_norm = (y1 - y0) / self.cfg.face_size
                    erp_box = reproject_cube_box(
                        face_idx=idx,
                        box_norm=(cx, cy, w_norm, h_norm),
                        crop_size=self.cfg.face_size,
                        erp_w=W,
                        erp_h=H,
                    )
                    if erp_box is not None:
                        dets.append(
                            {
                                "box": erp_box,
                                "score": float(score),
                                "phrase": phrase,
                            }
                        )

        # ─── NFOV MODE ─────────────────────────────────────────────
        elif mode == "nfov":
            crops = extract_nfov_crops(
                frame_bgr,
                self.cfg.nfov_fov_deg,
                self.cfg.nfov_stride_yaw,
                self.cfg.nfov_out_hw,
                pitch_angles=self.cfg.nfov_pitch_angles,
                yaw_offsets=self.cfg.nfov_yaw_offset,
                include_seam=self.cfg.nfov_include_seam,
            )
            images = [
                Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                for _, _, _, c in crops
            ]
            keys = [(yaw, pitch, fov) for yaw, pitch, fov, _ in crops]

            if len(images) > 1:
                logger.debug(
                    f"ERPBoxDetector: batch detecting {len(images)} NFoV crops"
                )
                boxes_list, scores_list, phrases_list = self.det_engine.detect_batch(
                    images,
                    labels=self.prompt,
                    box_threshold=self.cfg.gdino_box_threshold,
                    text_threshold=self.cfg.gdino_text_threshold,
                )
            else:
                logger.debug("ERPBoxDetector: single NFoV detection")
                raw = self.det_engine.detect(
                    images[0],
                    labels=self.prompt,
                    box_threshold=self.cfg.gdino_box_threshold,
                    text_threshold=self.cfg.gdino_text_threshold,
                )
                boxes_list = [raw[0]]
                scores_list = [raw[1]]
                phrases_list = [raw[2]]

            for (yaw, pitch, fov), boxes, scores, phrases in zip(
                keys, boxes_list, scores_list, phrases_list
            ):
                # filter & ensure tensor types
                boxes, scores, phrases = filter_face_detections(
                    boxes,
                    scores,
                    phrases,
                    self.cfg.nfov_out_hw[1],
                    self.cfg.nfov_out_hw[0],
                    max_face_frac=0.9,
                    min_score=self.cfg.gdino_box_threshold,
                )
                if not torch.is_tensor(boxes):
                    boxes = torch.tensor(boxes, dtype=torch.float32)
                if not torch.is_tensor(scores):
                    scores = torch.tensor(scores, dtype=torch.float32)

                # reproject each
                crop_size = self.cfg.nfov_out_hw[1]
                for (x0, y0, x1, y1), score, phrase in zip(
                    boxes.cpu().numpy(),
                    scores.cpu().numpy(),
                    phrases,
                ):
                    cx = (x0 + x1) / 2 / crop_size
                    cy = (y0 + y1) / 2 / crop_size
                    w_norm = (x1 - x0) / crop_size
                    h_norm = (y1 - y0) / crop_size
                    erp_box = project_nfov_box(
                        box_norm=(cx, cy, w_norm, h_norm),
                        yaw=float(yaw),
                        pitch=float(pitch),
                        crop_size=crop_size,
                        erp_w=W,
                        erp_h=H,
                        fov_deg=fov,
                    )
                    dets.append(
                        {
                            "box": erp_box,
                            "score": float(score),
                            "phrase": phrase,
                        }
                    )

        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

        # ─── POST‐PROCESS ────────────────────────────────────────────
        if mode == "nfov":
            # dets = spherical_clustering(
            #     dets,
            #     W,
            #     H,
            #     eps_degrees=self.cfg.eps_degrees,
            # )
            dets = agglomerative_spherical_nms(
                dets,
                W,
                H,
            )
        else:
            dets = hard_spherical_nms(
                dets=dets,
                iou_thresh=self.cfg.nms_iou_thresh,
                erp_h=H,
                erp_w=W,
            )
        dets = filter_seam_artifacts(
            dets,
            fw=W,
            fh=H,
            max_w_frac=self.cfg.seam_max_w_frac,
            max_h_frac=self.cfg.seam_max_h_frac,
        )

        return dets


class ERPFullPipeline:
    """
    End-to-end ERP annotation pipeline:
      - "full" mode: detect+annotate directly on the ERP frame
      - "cubemap" mode: detect+annotate on each cube face, then reproject back

    Uses GroundingDINO for box detection (batched when multiple tiles),
    optional SAM for mask refinement, and the ERP Warp utilities
    for cubemap splitting & recomposition.
    """

    def __init__(
        self,
        det_engine: GroundingDINOEngine,
        cfg: PipelineConfig,
        prompt: List[str],
        sam_client: Optional[SAMClient] = None,
    ):
        """
        Args:
            det_engine:   Preconfigured detection engine.
            cfg:         PipelineConfig with face_size, reproj_mode, etc.
            prompt:      Text prompt for DINO.
            sam_client:  Optional SAMClient for mask overlay.
        """
        self.det_engine = det_engine
        self.cfg = cfg
        self.prompt = prompt
        self.sam_client = sam_client

    @timeit
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        mode: str,
    ) -> Tuple[List[Dict], Image.Image]:
        """
        Run detection+annotation on one ERP frame, then reproject back.

        Args:
            frame_bgr: H×W×3 BGR input frame.
            mode:      "full" or "cube" (must match cfg.mode).

        Returns:
            dets:      List of dicts with keys {box, score, phrase}.
            result:    PIL.Image: annotated ERP panorama (RGB).
        """
        H, W = frame_bgr.shape[:2]

        # 1) Convert BGR→PIL for detection
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # ─── cubemap branch ───────────────────────────────────────
        if mode == "cube":
            # split into all six faces
            faces: Dict[str, Image.Image] = erp_to_cubemap(
                frame_pil, self.cfg.face_size
            )
            # start annotated_tiles as a full copy of these six faces
            annotated_tiles: Dict[str, Image.Image] = faces.copy()

            # only run detection on the faces defined in cfg.cube_faces
            items: List[Tuple[str, Image.Image]] = [
                (FACE_NAMES[idx], faces[FACE_NAMES[idx]]) for idx in self.cfg.cube_faces
            ]

        # ─── full‐frame branch ─────────────────────────────────────
        else:
            # use key "full" for single‐tile detection
            annotated_tiles = {"full": frame_pil}
            items = [("full", frame_pil)]

        # 2) Extract just the images for batch vs single detection
        images = [tile for _, tile in items]

        # 3) Run detection: batch if multiple tiles, else single
        if len(images) > 1:
            logger.debug(
                f"ERPFullPipeline: running batch detect on {len(images)} tiles"
            )
            raw_boxes_list, raw_scores_list, raw_phrases_list = (
                self.det_engine.detect_batch(
                    images,
                    labels=self.prompt,
                    box_threshold=self.cfg.gdino_box_threshold,
                    text_threshold=self.cfg.gdino_text_threshold,
                )
            )
        else:
            logger.debug(f"ERPFullPipeline: running single detect on {mode} frame")
            raw_boxes, raw_scores, raw_phrases = self.det_engine.detect(
                images[0],
                labels=self.prompt,
                box_threshold=self.cfg.gdino_box_threshold,
                text_threshold=self.cfg.gdino_text_threshold,
            )
            raw_boxes_list = [raw_boxes]
            raw_scores_list = [raw_scores]
            raw_phrases_list = [raw_phrases]

        # 4) Annotate each selected tile & collect reprojections
        dets: List[Dict] = []
        for (key, tile), raw_boxes, raw_scores, raw_phrases in zip(
            items, raw_boxes_list, raw_scores_list, raw_phrases_list
        ):
            # 4a) filter out giant / low‐score
            boxes, scores, phrases = filter_face_detections(
                raw_boxes,
                raw_scores,
                raw_phrases,
                tile.width,
                tile.height,
                max_face_frac=0.95,  # Changed from 0.9 to allow larger objects
                min_score=self.cfg.gdino_box_threshold,
            )

            # ensure tensor for uniformity
            if not torch.is_tensor(boxes):
                boxes = torch.tensor(boxes, dtype=torch.float32)
            if not torch.is_tensor(scores):
                scores = torch.tensor(scores, dtype=torch.float32)

            # 4b) optional SAM mask overlay
            tile_np = np.array(tile)  # RGB
            annotated_np = tile_np.copy()
            if self.sam_client and len(boxes):
                self.sam_client.set_image(tile)
                for box in boxes:
                    coords = (
                        box.cpu().tolist()
                        if isinstance(box, torch.Tensor)
                        else list(box)
                    )
                    # return multiple masks + their scores
                    mask_batch, mask_scores, _ = self.sam_client.predict(
                        box=coords, multimask_output=True
                    )
                    # log all candidates
                    # for i, score in enumerate(mask_scores):
                    #     logger.debug(
                    #         f"SAM mask candidate #{i} for box {coords} → score = {score:.3f}"
                    #     )

                    # combine all masks above threshold
                    thresh = getattr(self.cfg, "sam_min_mask_iou", 0.8)
                    good_masks = []
                    for mask, score in zip(mask_batch, mask_scores):
                        if score >= thresh:
                            if not isinstance(mask, np.ndarray):
                                mask = mask.cpu().numpy()
                            good_masks.append(mask.astype(bool))

                    if not good_masks:
                        logger.debug(f"→ no SAM masks ≥ {thresh:.3f} for box {coords}")
                    else:
                        # logger.debug(
                        #     f"→ merging {len(good_masks)} SAM masks ≥ {thresh:.3f} for box {coords}"
                        # )
                        combined = np.any(np.stack(good_masks, axis=0), axis=0)
                        
                        # Improve mask quality: fill small holes and smooth boundaries
                        from scipy import ndimage
                        try:
                            # Fill small holes in the mask
                            combined = ndimage.binary_fill_holes(combined)
                            # Apply morphological closing to connect nearby regions
                            kernel = np.ones((3, 3), np.uint8)
                            combined = ndimage.binary_closing(combined, structure=kernel)
                            # Apply slight dilation to ensure complete coverage
                            combined = ndimage.binary_dilation(combined, structure=kernel, iterations=1)
                        except ImportError:
                            # If scipy is not available, use basic operations
                            logger.warning("scipy not available, using basic mask operations")
                        
                        annotated_np[combined] = [255, 0, 0]
                        # logger.debug(f"→ applied improved combined SAM mask for box {coords}")

            # 4c) skip drawing boxes & labels - keep only red mask overlay
            # Just convert the image with red mask overlay to PIL format
            annotated_tiles[key] = Image.fromarray(annotated_np)

            # 4d) reproject each filtered box → ERP coords
            for (x0, y0, x1, y1), sc, ph in zip(
                boxes.cpu().numpy(), scores.cpu().numpy(), phrases
            ):
                if key == "full":
                    px_box = np.array([int(x0), int(y0), int(x1), int(y1)], float)
                else:
                    face_idx = FACE_NAMES.index(key)
                    cx = (x0 + x1) / 2 / tile.width
                    cy = (y0 + y1) / 2 / tile.height
                    w_norm = (x1 - x0) / tile.width
                    h_norm = (y1 - y0) / tile.height
                    px_box = reproject_cube_box(
                        face_idx=face_idx,
                        box_norm=(cx, cy, w_norm, h_norm),
                        crop_size=self.cfg.face_size,
                        erp_w=W,
                        erp_h=H,
                    )
                if px_box is not None:
                    dets.append({"box": px_box, "score": float(sc), "phrase": ph})

        # 5) post‐process all reprojections
        # dets = hard_nms(dets, self.cfg.nms_iou_thresh)
        dets = hard_spherical_nms(
            dets=dets,
            iou_thresh=self.cfg.nms_iou_thresh,
            erp_h=H,
            erp_w=W,
        )
        dets = filter_seam_artifacts(
            dets,
            fw=W,
            fh=H,
            max_w_frac=self.cfg.seam_max_w_frac,
            max_h_frac=self.cfg.seam_max_h_frac,
        )

        # 6) finally reproject the full six annotated faces back to ERP
        if mode == "cube":
            result = cubemap_to_erp(
                faces=annotated_tiles,
                face_size=self.cfg.face_size,
                erp_h=H,
                erp_w=W,
                mode="bilinear",
            )
        else:
            result = annotated_tiles["full"]

        return dets, result
