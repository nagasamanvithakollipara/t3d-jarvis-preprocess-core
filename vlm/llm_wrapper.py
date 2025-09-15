import json
from PIL import Image

from typing import Optional, List, Dict, Union


from common.logger import logger
from core.models.ollama_client import OllamaClient
from core.models.openai_client import OpenAIClient
from core.models.qwen_vl_client import QwenVLClient
from core.utils.pipeline_utils import (
    strip_json_fences,
)

from core.config import PipelineConfig


class LLMClientWrapper:
    """Dispatch either to OllamaClient.generate or OpenAIClient.chat."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.ollama_client: Optional[OllamaClient] = None
        self.openai_client: Optional[OpenAIClient] = None
        self.qwen_client: Optional[QwenVLClient] = None

        if cfg.llm.lower() == "ollama":
            self.ollama_client = OllamaClient(model_name=cfg.ollama_model)
        elif cfg.llm.lower() == "qwen":
            # instantiate native HF Qwen2.5-VL client
            self.qwen_client = QwenVLClient(
                model_name=cfg.qwen_model,
                torch_dtype="auto",  # Optional: torch_dtype="bfloat16"
            )
        else:
            self.openai_client = OpenAIClient(model_name=cfg.openai_model)

    def _encode_img(self, img: Image.Image) -> str:
        """
        Encode a PIL Image to base64 JPEG.
        """
        import io
        import base64

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate(self, prompt: str, img: Optional[Image.Image] = None) -> str:
        """
        Single prompt+image inference. Blocks until response returns.
        """
        # ollama_client is not None
        if self.ollama_client is not None:
            resp = self.ollama_client.generate(
                prompt=prompt,
                images=[img] if img is not None else None,
            )
            raw = resp.get("response", "").strip()
            return strip_json_fences(raw)

        # qwen_client is not None
        if self.qwen_client is not None:
            if img is None:
                raise ValueError("Qwen-VL requires an image input")
            # single text+image via HuggingFace-backed QwenVLClient
            raw = self.qwen_client.generate(
                text=prompt,
                image=img,
            ).strip()
            return strip_json_fences(raw)

        # openai_client is not None
        if self.openai_client is not None:
            messages = [{"role": "user", "content": prompt}]
            resp = self.openai_client.generate(
                messages=messages,
                images=[img] if img is not None else None,
                temperature=0.2,
            )
            raw = resp.get("content", "").strip()
            return strip_json_fences(raw)

        # neither client was set
        raise RuntimeError("No LLM client configured in LLMClientWrapper")

    def generate_parallel(
        self,
        prompts: List[str],
        imgs: List[Image.Image],
        max_workers: Optional[int] = None,
    ) -> List[str]:
        """
        Parallel synchronous inference over prompt/image pairs.

        Args:
            prompts:    List of text prompts.
            imgs:       Corresponding list of PIL Images.
            max_workers:
                Maximum threads to spin up. If None, uses:
                self.cfg.ollama_concurrency for Ollama or
                self.cfg.openai_concurrency for OpenAI.

        Returns:
            List of response strings (one per prompt/image).
        """
        # Ollama parallel synchronous
        if self.ollama_client is not None:
            workers = max_workers or self.cfg.ollama_concurrency
            raw_list = self.ollama_client.generate_batch(
                prompts,
                [[img] for img in imgs],
                max_workers=workers,
            )
            return [strip_json_fences(r.get("response", "").strip()) for r in raw_list]

        # Qwen parallel synchronous
        if self.qwen_client is not None:
            workers = max_workers or self.cfg.qwen_concurrency

            raw = self.qwen_client.generate_batch(
                prompts,
                imgs,
                max_workers=workers,
            )
            return [strip_json_fences(r.strip()) if r is not None else "" for r in raw]

        # OpenAI parallel synchronous
        if self.openai_client is not None:
            workers = max_workers or self.cfg.openai_concurrency
            # build messages & image lists
            msgs_list: List[List[Dict[str, str]]] = [
                [{"role": "user", "content": p}] for p in prompts
            ]
            imgs_list: List[List[Union[str, Image.Image]]] = [[img] for img in imgs]
            raw_list = self.openai_client.generate_batch(
                messages_list=msgs_list,
                images_list=imgs_list,
                temperature=0.2,
                max_workers=workers,
            )
            return [strip_json_fences(r.get("content", "").strip()) for r in raw_list]

        raise RuntimeError("No LLM client configured in LLMClientWrapper")

    def generate_openai_batch(
        self,
        prompts: List[str],
        imgs: List[Image.Image],
        temperature: float = 0.2,
    ) -> List[str]:
        """
        One-shot asynchronous Batch API via OpenAI. Uploads all tasks,
        polls until completion, then returns list of response contents.
        """
        if self.openai_client is None:
            raise RuntimeError("OpenAI client required for batch mode")

        try:
            raw_objs = self.openai_client.batch_api(
                prompts=prompts,
                imgs=imgs,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"OpenAI Batch API failed: {e}")
            raise

        captions: List[str] = []
        for item in raw_objs:
            resp = item.get("response", {})
            body = resp.get("body", {})
            choices = body.get("choices") or []
            if choices and isinstance(choices, list):
                content = choices[0].get("message", {}).get("content", "")
            else:
                # fallback to full item dump on error
                content = json.dumps(item)
            captions.append(strip_json_fences(content.strip()))

        return captions
