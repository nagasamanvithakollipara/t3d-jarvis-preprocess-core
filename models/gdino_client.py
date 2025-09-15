import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Union, Sequence, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from groundingdino.util.inference import Model as GDModel
from common.logger import logger


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GroundingDINOEngine:
    """
    A simplified GroundingDINO engine that:
      - Only accepts a List[str] of classes per image
      - Uses the new predict_with_classes API
      - Runs sequentially on GPU/MPS or in parallel threads on CPU
    """

    def __init__(
        self,
        cfg_path: str,
        weights_path: str,
        cpu_workers: int = 4,
        gpu_batch_size: int = 8,
    ):
        """
        Args:
            cfg_path:     path to the model config (.yaml)
            weights_path: path to the model checkpoint (.pth)
            cpu_workers:  number of threads for CPU inference
        """
        self.device = select_device()
        self.predictor = GDModel(
            model_config_path=cfg_path,
            model_checkpoint_path=weights_path,
            device=self.device.type,
        )
        self.cpu_workers = cpu_workers
        self.gpu_batch_size = gpu_batch_size
        logger.info(f"GroundingDINOEngine initialized on {self.device}")

    def _tensor_to_rgb(self, t: torch.Tensor) -> np.ndarray:
        # assume C×H×W, normalized by ImageNet mean/std
        t = t.detach().cpu()
        if t.ndim != 3:
            raise TypeError(f"Tensor must be C×H×W, got {t.shape}")

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # (H×W×C) in [0..255]
        rgb = (t.permute(1, 2, 0).numpy() * std + mean) * 255.0

        return rgb.clip(0, 255).astype(np.uint8)

    def _to_bgr(
        self, frame: Union[str, Image.Image, np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Convert any supported input into a uint8 BGR NumPy array.
        """
        if isinstance(frame, str):
            img = Image.open(frame).convert("RGB")
            arr = np.array(img)
        elif isinstance(frame, Image.Image):
            arr = np.array(frame)
        elif isinstance(frame, torch.Tensor):
            arr = self._tensor_to_rgb(frame)
        elif isinstance(frame, np.ndarray):
            arr = frame
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] != 3:
            raise ValueError(f"Expected 3 channels, got shape {arr.shape}")

        if arr.dtype != np.uint8:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

        return arr[:, :, ::-1]  # RGB→BGR

    def detect(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        labels: List[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Single-image inference. Accepts:
          - image: path, PIL, np.ndarray, or preprocessed tensor
          - classes: list of class names (List[str])
        Returns:
          - boxes:   (M,4) tensor of [x0,y0,x1,y1]
          - scores:  (M,) tensor of confidences
          - phrases: list of M class names corresponding to each box
        """
        img_bgr = self._to_bgr(image)

        dets = self.predictor.predict_with_classes(
            image=img_bgr,
            classes=labels,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        boxes = torch.from_numpy(dets.xyxy)
        scores = torch.from_numpy(dets.confidence)

        # handle possible None
        ids = dets.class_id if dets.class_id is not None else []
        phrases = [labels[i] for i in ids]

        return boxes, scores, phrases

    def detect_caption(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Single-image, free‑form caption inference.
          image:          path / PIL / np.ndarray / preprocessed tensor
          caption:        arbitrary text prompt
          box_threshold:  min box‑confidence
          text_threshold: min text‑token confidence

        Returns:
          boxes:   (M,4) tensor of [x0,y0,x1,y1]
          scores:  (M,) tensor of confidence scores
          phrases: list of M raw phrases extracted
        """
        img_bgr = self._to_bgr(image)

        # This returns (Detections, List[str])
        dets, phrases = self.predictor.predict_with_caption(
            image=img_bgr,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        boxes = torch.from_numpy(dets.xyxy)  # (M,4)
        scores = torch.from_numpy(dets.confidence)  # (M,)

        return boxes, scores, phrases

    def _detect_batch_gpu(
        self,
        images: Sequence[Union[str, Image.Image, np.ndarray, torch.Tensor]],
        classes: List[str],
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
        """
        GPU/MPS path: sub‐batch inputs by self.gpu_batch_size, but within each
        sub‐batch call predict_with_classes per image.
        """
        all_bboxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_phrases: List[List[str]] = []

        # process in chunks
        for start in range(0, len(images), self.gpu_batch_size):
            chunk = images[start : start + self.gpu_batch_size]

            # convert and send each image to GPU
            bgrs = [self._to_bgr(img) for img in chunk]
            for bgr in bgrs:
                dets = self.predictor.predict_with_classes(
                    image=bgr,
                    classes=classes,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                # unpack into tensors & clean phrases
                boxes = torch.from_numpy(dets.xyxy)
                scores = torch.from_numpy(dets.confidence)

                ids = dets.class_id
                if ids is None:
                    id_list: List[int] = []
                else:
                    id_list = ids.tolist()
                phrases = [classes[i] for i in id_list]

                all_bboxes.append(boxes)
                all_scores.append(scores)
                all_phrases.append(phrases)

        return all_bboxes, all_scores, all_phrases

    def detect_batch(
        self,
        images: Sequence[Union[str, Image.Image, np.ndarray, torch.Tensor]],
        labels: List[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
        """
        Multi-image inference:
          - On GPU/MPS: runs in sequence via _detect_batch_gpu
          - On CPU: runs in parallel threads via _detect_batch_cpu
        """
        # GPU/MPS → simple sequential loop
        if self.device.type in ("cuda", "mps"):
            return self._detect_batch_gpu(images, labels, box_threshold, text_threshold)
        # CPU → thread‐pooled
        else:
            results: List[Any] = [None] * len(images)

            with ThreadPoolExecutor(max_workers=self.cpu_workers) as pool:
                future_to_idx = {
                    pool.submit(
                        self.detect, img, labels, box_threshold, text_threshold
                    ): idx
                    for idx, img in enumerate(images)
                }
                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    results[idx] = fut.result()

            bboxes, scores, phrases = zip(*results)
            return list(bboxes), list(scores), list(phrases)

    def detect_batch_caption(
        self,
        images: Sequence[Union[str, Image.Image, np.ndarray, torch.Tensor]],
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
        """
        Batch free‑form caption inference.

        - On GPU/MPS: loops through images sequentially (still using the GPU each call).
        - On CPU:    parallel threads to maximize throughput.
        """
        # GPU/MPS → simple sequential loop
        if self.device.type in ("cuda", "mps"):
            results = [
                self.detect_caption(img, caption, box_threshold, text_threshold)
                for img in images
            ]
        # CPU → thread‐pooled
        else:
            results: List[Any] = [None] * len(images)

            with ThreadPoolExecutor(max_workers=self.cpu_workers) as pool:
                futures = {
                    pool.submit(
                        self.detect_caption, img, caption, box_threshold, text_threshold
                    ): idx
                    for idx, img in enumerate(images)
                }
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()

        # unzip into three parallel lists
        bboxes_list, scores_list, phrases_list = zip(*results)
        return list(bboxes_list), list(scores_list), list(phrases_list)
