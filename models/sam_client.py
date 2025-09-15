import torch
import numpy as np
from typing import Optional, List, Tuple, Union, Any
from PIL import Image
from common.logger import logger
from core.config import PipelineConfig


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("MPS built:", torch.backends.mps.is_built())
        print("MPS available:", torch.backends.mps.is_available())
        return torch.device("mps")
    return torch.device("cpu")


class SAMClient:
    """
    Unified client for SAM v1 and SAM v2 segmentation models.

    Chooses implementation based on `cfg.sam_version`:
      - "v1": uses `segment_anything` (ViT backbones)
      - "v2": uses `sam2` repository (ResNet-Transformer hybrid)

    Usage:
        cfg = PipelineConfig()
        cfg.sam_version = "v2"
        sam = SAMClient(cfg)
        sam.set_image(pil_image)
        masks, scores, logits = sam.predict(box=np.array([x0,y0,x1,y1]))
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        version = cfg.sam_version.lower()
        self.device = select_device()

        if version == "v1":
            # Import SAM v1 API
            from segment_anything import (
                SamPredictor,
                sam_model_registry,
            )

            model_type = cfg.sam1_model_type
            checkpoint = cfg.sam_checkpoint_v1

            # Instantiate v1 model (e.g., vit_h) with checkpoint
            self.model = sam_model_registry[model_type](checkpoint=checkpoint)
            self.model.to(self.device)
            self.model.eval()

            self.predictor: Any = SamPredictor(self.model)
            self.version = "v1"
            logger.info(
                f"SAMClient: initialized SAM v1 on {self.device} with checkpoint {cfg.sam_checkpoint_v1}"
            )

        elif version == "v2":
            # Import SAM v2 API
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Build SAM-2 model & predictor
            self.model = build_sam2(cfg.sam2_cfg, cfg.sam2_checkpoint, device="cpu")
            self.model.to(self.device)
            self.model.eval()

            self.predictor: Any = SAM2ImagePredictor(self.model, device=self.device)
            self.version = "v2"
            logger.info(
                f"SAMClient: initialized SAM v2 with config [{cfg.sam2_cfg}] and checkpoint [{cfg.sam2_checkpoint}]"
            )
            logger.info(f"SAM v2 initialized on {self.device}")

        else:
            raise ValueError(
                f"Unsupported sam_version '{cfg.sam_version}'; choose 'v1' or 'v2'"
            )

    def set_image(self, img: Union[str, np.ndarray, Image.Image]):
        """
        Load an image into the SAM predictor.

        Args:
            img: File path, NumPy array (H×W×3), or PIL.Image.
        """
        # Always convert to RGB NumPy array
        if isinstance(img, str):
            img = np.array(Image.open(img).convert("RGB"))
        elif isinstance(img, Image.Image):
            img = np.array(img.convert("RGB"))
        elif isinstance(img, np.ndarray):
            img = np.ascontiguousarray(img)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        # Set image for predictor
        self.predictor.set_image(img)

    def predict(
        self,
        box: Optional[Union[np.ndarray, List[float]]] = None,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        multimask_output: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run segmentation with the current image and prompts.

        Args:
            box:             Optional [x0, y0, x1, y1] prompt as numpy or list.
            point_coords:    Optional (N×2) array of (x,y) prompts.
            point_labels:    Optional (N,) array of labels (1=fg,0=bg).
            multimask_output: If True, return multiple mask proposals.

        Returns:
            masks:  (M, H, W) boolean mask array.
            scores: (M,) float array of mask confidence.
            logits: (M, H, W) float array of mask logits.
        """
        # Prepare prompt inputs
        inputs: dict = {}
        if box is not None:
            box_np = (
                box if isinstance(box, np.ndarray) else np.asarray(box, dtype=float)
            )
            inputs["box"] = box_np
        if point_coords is not None and point_labels is not None:
            inputs["point_coords"] = point_coords
            inputs["point_labels"] = point_labels

        # Run prediction; both SAM v1 and v2 support this signature
        masks, scores, logits = self.predictor.predict(
            multimask_output=multimask_output,
            **inputs,
        )
        return masks, scores, logits
