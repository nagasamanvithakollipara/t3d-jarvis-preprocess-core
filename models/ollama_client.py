# models/ollama_client.py

"""HTTP client for Ollama-hosted vision models (e.g. Qwen2.5-VL)."""


import os
import requests
import base64
from typing import List, Optional, Dict, Any, Union
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()


class OllamaClient:
    """
    Generic HTTP client for an Ollama server.

    Args:
        host:  Base URL of Ollama (defaults to OLLAMA_HOST or http://localhost:11434).
        model: Model identifier (e.g. "qwen2.5vl:latest").

    Methods:
        generate(prompt, images=None, stream=False, options=None) → Dict
        Note: accepts either file‐paths *or* PIL.Image in the `images` list.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        model_name: str = "qwen2.5vl:latest",
    ) -> None:
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip(
            "/"
        )
        self.model = model_name

    def _encode_image_file(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _encode_image_obj(self, img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate(
        self,
        prompt: str,
        images: Optional[List[Union[str, Image.Image]]] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[Union[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Sends a generation request to Ollama.

        `images` may be a list of:
          - local file path (str)
          - PIL.Image.Image
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
        }
        if images:
            b64_list = []
            for i in images:
                if isinstance(i, str):
                    b64_list.append(self._encode_image_file(i))
                elif isinstance(i, Image.Image):
                    b64_list.append(self._encode_image_obj(i))
                else:
                    raise TypeError(f"Unsupported image type {type(i)}")
            payload["images"] = b64_list
        if options:
            payload["options"] = options
        if keep_alive is not None:
            # send negative keep_alive as a string to avoid the hang
            payload["keep_alive"] = (
                str(keep_alive) if isinstance(keep_alive, int) else keep_alive
            )

        r = requests.post(f"{self.host}/api/generate", json=payload)
        try:
            r.raise_for_status()
        except requests.HTTPError:
            print("STATUS:", r.status_code)
            print("BODY:", r.text)
            raise
        return r.json()

    def generate_batch(
        self,
        prompts: List[str],
        images_list: Optional[List[List[Union[str, Image.Image]]]] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        max_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Concurrently call generate() for each prompt+images pair.
        """
        from tqdm import tqdm

        def _worker(idx: int) -> Dict[str, Any]:
            imgs = images_list[idx] if images_list else None
            return self.generate(
                prompt=prompts[idx],
                images=imgs,
                stream=stream,
                options=options,
                keep_alive="30m",
            )

        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = [exe.submit(_worker, i) for i in range(len(prompts))]
            for future in tqdm(
                futures, total=len(futures), desc=f"ollama {self.model} parallel"
            ):
                results.append(future.result())

        return results
