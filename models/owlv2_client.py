import cv2
import torch
from PIL import Image
import numpy as np
from typing import List, Tuple, Union, Sequence, Optional

from transformers.models.owlvit import OwlViTProcessor
from transformers.models.owlv2 import (
    # Owlv2Processor,
    Owlv2ForObjectDetection,
)
from common.logger import logger


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class OWLv2Engine:
    """Zero-shot detection with google/owlv2-base-patch16-ensemble"""

    def __init__(
        self,
        model_name: str = "google/owlv2-base-patch16-ensemble",
        use_fast: bool = True,
        gpu_batch_size: int = 4,
    ) -> None:
        self.device = select_device()
        self.processor = OwlViTProcessor.from_pretrained(model_name, use_fast=use_fast)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)  # type: ignore

        if self.device.type == "cuda":
            self.model.half()
        self.model.eval()

        # default chunk size for GPU/MPS
        self.gpu_batch_size = gpu_batch_size

        logger.info(
            f"OWLv2Engine initialized with model [{model_name}] on {self.device}\n"
        )

    def _to_pil(
        self, img: Union[str, torch.Tensor, np.ndarray, Image.Image]
    ) -> Image.Image:
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255, 0, 255).astype("uint8")
            return Image.fromarray(arr)
        if isinstance(img, np.ndarray):
            arr = img
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(arr)
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        return img.convert("RGB")

    def detect(
        self,
        image: Union[str, torch.Tensor, np.ndarray, Image.Image],
        labels: List[str],
        threshold: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        pil = self._to_pil(image)

        # prepare single-item list for grounded detection
        text_labels = [labels]
        inputs = self.processor(
            text=text_labels,
            images=pil,
            return_tensors="pt",
            padding=True,
        )  # type: ignore
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(  # type: ignore
            outputs=outputs,
            target_sizes=[pil.size[::-1]],
            threshold=threshold,
            text_labels=text_labels,
        )
        res = results[0]
        boxes = res["boxes"].cpu()
        scores = res["scores"].cpu()
        phrases = res["text_labels"]  # already string labels

        return boxes, scores, phrases

    def detect_batch(
        self,
        images: Sequence[Union[str, torch.Tensor, np.ndarray, Image.Image]],
        labels: List[str],
        threshold: float = 0.1,
        batch_size: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
        """
        Batch zero-shot detection. Processes images in chunks of size
        (batch_size or self.gpu_batch_size) to balance throughput and memory.
        """
        bs = (
            batch_size
            if (batch_size is not None and batch_size > 0)
            else self.gpu_batch_size
        )
        pil_images = [self._to_pil(img) for img in images]

        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_phrases: List[List[str]] = []

        for i in range(0, len(pil_images), bs):
            chunk = pil_images[i : i + bs]
            text_labels = [labels] * len(chunk)

            enc = self.processor(
                text=text_labels,
                images=chunk,
                return_tensors="pt",
                padding=True,
            )  # type: ignore
            inputs = {
                k: v.to(self.device) for k, v in enc.items() if torch.is_tensor(v)
            }

            with torch.inference_mode():
                outputs = self.model(**inputs)

            sizes = [img.size[::-1] for img in chunk]
            results = self.processor.post_process_grounded_object_detection(  # type: ignore
                outputs=outputs,
                target_sizes=sizes,
                threshold=threshold,
                text_labels=text_labels,
            )

            for res in results:
                all_boxes.append(res["boxes"].cpu())
                all_scores.append(res["scores"].cpu())
                all_phrases.append(res["text_labels"])

        return all_boxes, all_scores, all_phrases
