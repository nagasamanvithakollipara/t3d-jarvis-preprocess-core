import torch
import os
import base64
from typing import Optional, List, Union, Sequence
from PIL import Image
from io import BytesIO

from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from huggingface_hub import InferenceClient
from common.logger import logger


class QwenVLClient:
    """
    Qwen2.5-VL client with two inference hosts:
      - local   : load into PyTorch (CUDA/MPS/CPU)
      - hf_api  : call the HF Inference API

    Args:
        model_name: HF model identifier (e.g. "Qwen/Qwen2.5-VL-7B-Instruct").
        host:       "local" or "hf_api"
        device:     torch device ("cuda","mps","cpu") for local
        torch_dtype:"auto", torch.float16, or torch.float32 for local
        hf_token:   HF API token
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        host: str = "hf_api",
        device: Optional[str] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        hf_token: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.host = host.lower()
        self.token = hf_token or os.getenv("HF_TOKEN")

        if self.host not in ("local", "hf_api"):
            raise ValueError("host must be 'local' or 'hf_api'")

        if self.host == "hf_api":
            # ——— initialize HF hosted client ———
            self.hf_client = InferenceClient(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                provider="hyperbolic",
                token=self.token,
            )
            logger.info(f"QwenVLClient using HF Inference API for {self.model_name}")
        else:
            # ——— local host setup ———
            # Device selection: prefer CUDA, then MPS (macOS), else CPU
            if device:
                self.device = device
            else:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"

            # Determine dtype
            if torch_dtype == "auto":
                if self.device in ("cuda", "mps"):
                    dtype = torch.float16
                else:
                    dtype = torch.float32
            else:
                dtype = (
                    torch_dtype
                    if isinstance(torch_dtype, torch.dtype)
                    else getattr(torch, str(torch_dtype))
                )

            logger.info(
                f"Initializing QwenVLClient on device={self.device} with dtype={dtype}"
            )

            # Load processor and model onto specified device
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                token=self.token,
                use_fast=True,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                token=self.token,
                torch_dtype=dtype,
                # device_map removed to avoid fork-related MPS issues
            )
            # Move model to the correct device using a torch.device object
            device_obj = torch.device(self.device)
            self.model = self.model.to(device_obj)  # type: ignore
            self.model.eval()

    def generate(
        self,
        text: str,
        image: Union[str, Image.Image],
        max_new_tokens: int = 256,
        **gen_kwargs,
    ) -> str:
        """
        Generate a response given a text prompt and one image, leveraging
        apply_chat_template to align <image> tokens with visual features.
        Returns the generated text (with special tokens stripped).
        """

        # ——— HF Inference API path ———
        if self.host == "hf_api":
            # encode image to base64
            if isinstance(image, str):
                with open(image, "rb") as f:
                    img_bytes = f.read()
            else:
                buf = BytesIO()
                image.convert("RGB").save(buf, format="JPEG")
                img_bytes = buf.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            # send chat completion request
            completion = self.hf_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            },
                        ],
                    }
                ],
            )
            # extract generated message
            msg = completion.choices[0].message
            return getattr(msg, "content", getattr(msg, "text", "")).strip()

        # ——— local path ———
        # 1) Load & normalize the image
        if isinstance(image, str):
            img_obj = Image.open(image).convert("RGB")
        else:
            img_obj = image.convert("RGB")

        # 2) Build a single-message chat payload
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_obj},
                    {"type": "text", "text": text},
                ],
            }
        ]

        # 3) Let the processor inject all the vision tokens & system/user tags:
        chat_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 4) Convert into model-ready tensors:
        inputs = self.processor(
            text=[chat_input],
            images=[img_obj],
            padding=True,
            return_tensors="pt",
        )
        # move everything to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 5) Run generation
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )

        # 6) Strip off the prompt tokens, leaving only the newly generated IDs
        input_len = inputs["input_ids"].shape[1]
        gen_ids = generated[0, input_len:]

        # 7) Decode the output IDs into a string
        output = self.processor.batch_decode(
            [gen_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output.strip()

    def generate_batch(
        self,
        prompts: List[str],
        images: Sequence[Union[str, Image.Image]],
        max_new_tokens: int = 256,
        max_workers: int = 4,
        **gen_kwargs,
    ) -> List[Optional[str]]:
        """
        Parallel version of .generate(), preserving robust error handling.
        Returns a list of strings or None for failures.
        """
        # ——— HF Inference API: purely serial, using only hf_client ———
        if self.host == "hf_api":
            return [
                self.generate(text, img, max_new_tokens=max_new_tokens, **gen_kwargs)
                for text, img in zip(prompts, images)
            ]

        # ——— local path ———
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: List[Optional[str]] = [None] * len(prompts)

        def _job(idx: int):
            try:
                results[idx] = self.generate(
                    prompts[idx],
                    images[idx],
                    max_new_tokens=max_new_tokens,
                    **gen_kwargs,
                )
            except Exception as e:
                logger.error(
                    f"Error in QwenVLClient.generate_batch idx={idx}: {e}",
                    exc_info=True,
                )
                results[idx] = None

        # spin up a thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_job, i) for i in range(len(prompts))]
            for _ in as_completed(futures):
                pass

        return results
