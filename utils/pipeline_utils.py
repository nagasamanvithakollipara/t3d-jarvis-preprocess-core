# core/utils/pipeline_utils.py

import time
import json
import re
from typing import Any


def strip_json_fences(text: str) -> str:
    """
    Remove ``` or ```json fences around JSON, capturing whatever’s inside—
    whether it starts with { or [—and return just the inner text.
    """
    # This will match ``` or ```json, then lazily capture any chars (including newlines)
    # up to the next ```
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(1) if m else text


def sanitize_vlm_json(text: str) -> str:
    """
    Heuristically correct common JSON formatting issues in VLM outputs,
    specifically around malformed "bbox" fields, so downstream json.loads()
    calls will succeed.

    Steps:
      1. Normalize any variant of the bbox key and its opening bracket
         (e.g. bbox[…], bbox=[…], or misplaced quotes) into the canonical `"bbox":[`.
      2. Remove stray quotation marks immediately following a closing bracket.
      3. Strip any trailing commas before a closing list or object delimiter.
    """
    corrected = text

    # 1) Normalize key + opening bracket:
    #    Matches things like bbox[…], "bbox"=[…], bbox "[…]", etc.
    #    and rewrites them to the valid `"bbox":[`
    corrected = re.sub(
        r'"?bbox[^[]*\[',
        '"bbox":[',
        corrected,
    )

    # 2) Remove stray quote right after a closing bracket:
    #    Turns `]\"` back into `]`
    corrected = re.sub(r'\]"', "]", corrected)

    # 3) Drop trailing commas before `]` or `}`:
    #    Ensures no trailing commas remain that would break JSON syntax.
    corrected = re.sub(r",\s*([\]\}])", r"\1", corrected)

    # — Add further sanitization rules here if new VLM quirks appear —

    return corrected


def safe_parse_json(text: str) -> Any:
    """
    Try a normal json.loads; on failure, sanitize and retry.
    Raises JSONDecodeError if it still fails.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = sanitize_vlm_json(text)
        return json.loads(cleaned)


def timeit(func):
    """
    Decorator to measure and print execution time of a function.
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[{func.__name__}] Total time: {elapsed:.2f}s")
        return result

    return wrapper
