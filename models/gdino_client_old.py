import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Union, Any, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from groundingdino.util.inference import load_model, preprocess_caption
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.inference import T

from common.logger import logger


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GroundingDINOEngine:
    """
    A unified GroundingDINO inference engine that:
      - On GPU/MPS: runs batched inference
      - On CPU: runs per-image inference in parallel threads
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
            cfg_path: path to your model config (.yaml)
            weights_path: path to your model weights (.pth)
            cpu_workers: number of threads to use when on CPU
            gpu_batch_size: max images per batch when on GPU/MPS
        """
        self.device = select_device()
        self.model = load_model(cfg_path, weights_path, device="cpu")
        self.model.to(self.device)
        self.model.eval()
        self.cpu_workers = cpu_workers
        self.gpu_batch_size = gpu_batch_size

        # set up torchvision-style preprocessing
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        logger.info(f"GroundingDINOEngine initialized on {self.device}\n")

    def _preprocess(
        self, frame: Union[str, Image.Image, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Accepts either:
         - file path (str)
         - a NumPy/OpenCV BGR array
         - a PIL.Image
         - already a Tensor (skips transforms)
        Returns a (C,H,W) float Tensor on CPU.
        """
        if isinstance(frame, str):
            frame = Image.open(frame).convert("RGB")

        # Already a tensor → assume it’s preprocessed
        elif isinstance(frame, torch.Tensor):
            return frame

        # NumPy array (RGB) → convert to PIL
        elif isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)

        # PIL image → nothing to do
        elif isinstance(frame, Image.Image):
            pass

        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")

        # Apply the same transforms (resize, to-tensor, normalize)
        img_tensor, _ = self.transform(frame, None)
        return img_tensor

    def detect(
        self,
        image: Union[str, Image.Image, torch.Tensor, np.ndarray],
        labels: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Single-image inference.  Returns:
          - boxes: (M,4) tensor of [x0,y0,x1,y1] in pixel coords
          - scores: (M,) tensor of max logits
          - phrases: list of M strings
        """
        # 1) grab original size in pixels
        if isinstance(image, np.ndarray):
            H, W = image.shape[:2]
        elif isinstance(image, Image.Image):
            W, H = image.size
        else:
            raise TypeError(
                f"detect() needs a NumPy array or PIL image, got {type(image)}"
            )

        # 2) preprocess & infer
        img_t = self._preprocess(image).unsqueeze(0).to(self.device)
        cap = preprocess_caption(labels)
        with torch.no_grad():
            outputs = self.model(img_t, captions=[cap])

        # 3) pull out raw logits & normalized boxes
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (N_queries, N_classes)
        raw_boxes = outputs["pred_boxes"].cpu()[0]  # (N_queries, 4) in cx,cy,w,h

        # 4) filter by box confidence
        scores_per_query = logits.max(dim=1)[0]  # max over classes
        keep = scores_per_query > box_threshold
        sel_logits = logits[keep]  # (M, N_classes)
        sel_boxes = raw_boxes[keep]  # (M, 4)

        # 5) convert to absolute [x0,y0,x1,y1] pixels
        cx, cy, bw, bh = sel_boxes.unbind(-1)
        x0 = (cx - 0.5 * bw) * W
        y0 = (cy - 0.5 * bh) * H
        x1 = (cx + 0.5 * bw) * W
        y1 = (cy + 0.5 * bh) * H
        abs_boxes = torch.stack([x0, y0, x1, y1], dim=-1)  # (M,4)

        # 6) extract phrases (unchanged)
        tokenized = self.model.tokenizer(cap)
        phrases = [
            get_phrases_from_posmap(
                (logit > text_threshold), tokenized, self.model.tokenizer
            ).replace(".", "")
            for logit in sel_logits
        ]

        # 7) final scores = max over text tokens
        final_scores = sel_logits.max(dim=1)[0]

        return abs_boxes, final_scores, phrases

    def detect_batch(
        self,
        images: Sequence[Union[str, Image.Image, torch.Tensor, np.ndarray]],
        labels: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
        """
        Multi-image inference. Automatically chooses:
          - True batching on GPU/MPS
          - Parallel per-image on CPU
        """
        if self.device.type in ("cuda", "mps"):
            return self._detect_batch_gpu(images, labels, box_threshold, text_threshold)
        else:
            return self._detect_batch_cpu(images, labels, box_threshold, text_threshold)

    def _detect_batch_gpu(
        self,
        images: Sequence[Union[str, Image.Image, torch.Tensor, np.ndarray]],
        labels: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
        """
        Multi-image detection on GPU/MPS: splits `images` into sub-batches of size
        `self.gpu_batch_size` and processes each on the device.

        Args:
            images: sequence of frames (file paths, arrays, PIL Images, or Tensors)
            labels: text prompt for grounding
            box_threshold: minimum box-confidence to keep a detection
            text_threshold: minimum text-confidence to extract a phrase

        Returns:
            A tuple of three lists (one entry per input image):
              - bboxes: list of Tensors of shape (Ni, 4)
              - scores: list of Tensors of shape (Ni,)
              - phrases: list of lists of strings
        """
        # 1) Build & move the full tensor batch to GPU
        tensors = torch.stack([self._preprocess(im) for im in images], dim=0).to(
            self.device
        )
        cap = preprocess_caption(labels)

        all_bboxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_phrases: List[List[str]] = []

        # 2) Process in sub-batches
        for start in range(0, len(images), self.gpu_batch_size):
            batch = tensors[start : start + self.gpu_batch_size]  # on GPU
            with torch.no_grad():
                outputs = self.model(batch, captions=[cap] * batch.size(0))

            logits = outputs["pred_logits"].sigmoid()  # (B, Nq, Nc)
            raw_boxes = outputs["pred_boxes"]  # (B, Nq, 4) normalized cx,cy,w,h
            mask = logits.max(dim=2)[0] > box_threshold  # (B, Nq)

            tokenized = self.model.tokenizer(cap)
            for i in range(batch.size(0)):
                sel = mask[i]
                if not sel.any():
                    all_bboxes.append(torch.empty((0, 4), dtype=torch.float32))
                    all_scores.append(torch.empty((0,), dtype=torch.float32))
                    all_phrases.append([])
                    continue

                sel_logits = logits[i][sel]  # (M, Nc)
                sel_boxes = raw_boxes[i][sel]  # (M, 4)

                frame = images[start + i]
                if isinstance(frame, np.ndarray):
                    H, W = frame.shape[:2]
                elif isinstance(frame, Image.Image):
                    W, H = frame.size
                elif isinstance(frame, torch.Tensor):
                    # assume (C,H,W)
                    _, H, W = frame.shape
                elif isinstance(frame, str):
                    img = Image.open(frame)
                    W, H = img.size
                else:
                    raise TypeError(f"Unsupported frame type: {type(frame)}")

                # convert normalized [cx,cy,w,h] → pixel [x0,y0,x1,y1]
                cx, cy, bw, bh = sel_boxes.unbind(-1)
                x0 = (cx - 0.5 * bw) * W
                y0 = (cy - 0.5 * bh) * H
                x1 = (cx + 0.5 * bw) * W
                y1 = (cy + 0.5 * bh) * H
                abs_boxes = torch.stack([x0, y0, x1, y1], dim=-1)

                # phrases & final score
                phrases = [
                    get_phrases_from_posmap(
                        (lg > text_threshold), tokenized, self.model.tokenizer
                    ).replace(".", "")
                    for lg in sel_logits
                ]
                scores = sel_logits.max(dim=1)[0]

                # push to CPU for annotation
                all_bboxes.append(abs_boxes.cpu())
                all_scores.append(scores.cpu())
                all_phrases.append(phrases)

        return all_bboxes, all_scores, all_phrases

    def _detect_batch_cpu(
        self,
        images: Sequence[Union[str, Image.Image, torch.Tensor, np.ndarray]],
        labels: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
        """
        Performs multi-image detection on CPU by dispatching single-image
        `detect` calls in parallel threads.

        Args:
            images: list of frames (file paths, OpenCV arrays, PIL Images, or Tensors)
            caption: text prompt to guide detection
            box_threshold: min box-confidence to keep a detection
            text_threshold: min text-confidence to extract a phrase

        Returns:
            A tuple of three lists, each of length len(images):
              - bboxes_list: list of (Nᵢ, 4) box Tensors per image
              - scores_list: list of (Nᵢ,) score Tensors per image
              - phrases_list: list of Nᵢ-length phrase lists per image
        """
        # Prepare a container to hold each image’s (boxes, scores, phrases)
        results: List[Any] = [None] * len(images)

        # Launch detect() in a threadpool to leverage multiple CPU cores
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

        # Unpack results into three parallel lists
        bboxes_list, scores_list, phrases_list = zip(*results)
        return list(bboxes_list), list(scores_list), list(phrases_list)
