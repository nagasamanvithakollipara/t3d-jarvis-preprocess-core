import torch
import cv2
import numpy as np
from typing import List, Union, Tuple, Sequence
from PIL import Image

from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.modeling_auto import AutoModelForZeroShotObjectDetection


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GroundingDINOHFEngine:
    def __init__(
        self,
        model: str = "IDEA-Research/grounding-dino-tiny",
    ) -> None:
        self.device = select_device()
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model).to(
            self.device
        )
        self.processor = AutoProcessor.from_pretrained(model)

    def _preprocess(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, np.ndarray):
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def detect(
        self,
        image: Union[str, Image.Image, np.ndarray],
        labels: List[str],
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        GroundingDINO-compatible single image inference.
        Returns: (boxes_tensor, scores_tensor, phrases_list)
        """
        pil_img = self._preprocess(image)
        text_labels = [labels]

        inputs = self.processor(
            images=pil_img, text=text_labels, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_size = [pil_img.size[::-1]]  # (H, W)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_size,
        )

        result = results[0]
        boxes = result["boxes"].detach().cpu()
        scores = result["scores"].detach().cpu()
        phrases = result["labels"]  # already a list of label strings

        return boxes, scores, phrases

    def detect_batch(
        self,
        images: Sequence[Union[str, Image.Image, np.ndarray]],
        labels: List[str],
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
        batch_size: int = 8,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
        """
        Batch inference using Grounding DINO. Returns lists of (boxes, scores, phrases) per image.
        """
        pil_images = [self._preprocess(img) for img in images]
        all_boxes, all_scores, all_phrases = [], [], []

        for i in range(0, len(pil_images), batch_size):
            batch_imgs = pil_images[i : i + batch_size]
            batch_texts = [labels] * len(batch_imgs)

            inputs = self.processor(
                images=batch_imgs, text=batch_texts, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = [img.size[::-1] for img in batch_imgs]
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )

            for result in results:
                boxes = result["boxes"].detach().cpu()
                scores = result["scores"].detach().cpu()
                phrases = result["labels"]
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_phrases.append(phrases)

        return all_boxes, all_scores, all_phrases
