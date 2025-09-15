# models/openai_client.py

"""Client for OpenAI’s GPT-4o multimodal chat endpoint."""

import os
import time
import json
import base64

from typing import List, Dict, Any, Optional, Union
from io import BytesIO
from PIL import Image
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv

load_dotenv()


class OpenAIClient:
    """
    Wrapper for the GPT-4o chat-completion API with image support.

    Args:
        api_key: OpenAI API key, or read from OPENAI_API_KEY env var.
        model: Model name (e.g. "gpt-4o").

    Methods:
        chat(messages, images=None, temperature=0.2, stream=False) -> Dict[str, Any]
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
    ) -> None:
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model_name

    def _encode_image(self, img: Union[str, Image.Image]) -> str:
        """Accepts a file path or PIL.Image, returns base64 JPEG."""
        if isinstance(img, str):
            with open(img, "rb") as f:
                data = f.read()
        else:
            buf = BytesIO()
            img.save(buf, format="JPEG")
            data = buf.getvalue()
        return base64.b64encode(data).decode("utf-8")

    def generate(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List[Union[str, Image.Image]]] = None,
        temperature: float = 0.2,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Synchronous single Chat Completion call.

        Args:
            messages: List of dicts with keys "role" and "content" (text prompt).
            images: Optional list of local image paths *or* PIL.Image in the `images` list.
            temperature: Sampling temperature (0.0–2.0).
            stream: Whether to request streaming output.

        Returns:
            The assistant’s reply as a dict, with `"content"` holding the response text
            (e.g. JSON with an `annotations` field if your prompt requests it).
        """
        content_blocks: List[Dict[str, Any]] = []
        for m in messages:
            content_blocks.append({"type": "text", "text": m["content"]})
        if images:
            for i in images:
                b64 = self._encode_image(i)
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )
        payload = [{"role": "user", "content": content_blocks}]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=payload,  # type: ignore
            temperature=temperature,
            stream=stream,
        )
        return resp.choices[0].message.to_dict()

    def generate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        images_list: Optional[List[List[Union[str, Image.Image]]]] = None,
        temperature: float = 0.2,
        stream: bool = False,
        max_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Parallel synchronous Chat calls using ThreadPoolExecutor.
        Each .chat() still blocks until its frame is done.
        Returns a list of the raw response dicts.
        """
        from tqdm import tqdm

        def _job(idx: int) -> Dict[str, Any]:
            return self.generate(
                messages=messages_list[idx],
                images=images_list[idx] if images_list else None,
                temperature=temperature,
                stream=stream,
            )

        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = [exe.submit(_job, i) for i in range(len(messages_list))]
            for future in tqdm(
                futures, total=len(futures), desc=f"openai {self.model} parallel"
            ):
                results.append(future.result())

        return results

    def batch_api(
        self,
        prompts: List[str],
        imgs: Optional[List[Image.Image]] = None,
        temperature: float = 0.2,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
    ) -> List[Dict[str, Any]]:
        """
        True asynchronous Batch API:
          1. Build a JSONL file of tasks
          2. client.files.create(..., purpose="batch")
          3. client.batches.create(input_file_id=..., endpoint="/v1/chat/completions")
          4. Poll client.batches.retrieve(...) until status == 'completed'
          5. client.files.content(output_file_id) → JSONL result
        Returns list of response dicts in original order.
        """
        # Step 1: Build tasks
        tasks: List[Dict[str, Any]] = []
        for idx, prompt in enumerate(prompts):
            blocks: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            if imgs:
                b64 = self._encode_image(imgs[idx])
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )
            tasks.append(
                {
                    "custom_id": str(idx),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "temperature": temperature,
                        # Wrap the content blocks in a proper messages array:
                        "messages": [{"role": "user", "content": blocks}],
                    },
                }
            )

        # Step 2: Write JSONL then upload
        with NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as tf:
            for t in tasks:
                tf.write(json.dumps(t) + "\n")
            tf.flush()
            batch_file = self.client.files.create(
                file=open(tf.name, "rb"), purpose="batch"
            )

        # Step 3: Create batch job
        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        # Step 4: Poll until done or timeout
        start = time.time()
        while True:
            bj = self.client.batches.retrieve(batch_job.id)
            if bj.status == "completed":
                break
            if time.time() - start > timeout:
                raise TimeoutError("OpenAI batch job timed out")
            time.sleep(poll_interval)

        # Step 5: Download results
        if bj.output_file_id:
            result_file = self.client.files.content(bj.output_file_id)
            data = result_file.content.decode("utf-8").splitlines()
            results = [json.loads(line) for line in data]
            results.sort(key=lambda x: int(x["custom_id"]))
            return results

        if bj.error_file_id:
            err_file = self.client.files.content(bj.error_file_id)
            err_lines = err_file.content.decode("utf-8").splitlines()
            errs = [json.loads(line) for line in err_lines]
            raise RuntimeError(
                "Batch API failed with errors:\n" + json.dumps(errs, indent=2)
            )

        # neither output nor error file!
        raise RuntimeError(
            "Batch completed with neither output_file_id nor error_file_id"
        )
