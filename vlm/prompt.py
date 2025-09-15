# core/config/prompt.py
from dataclasses import dataclass, field


@dataclass
class PromptTemplates:
    """All the templated prompts used across the pipelines."""

    vlm_detection_gpt4o_prompt: str = field(
        default=(
            "You are an object detector.  The image coordinates are defined as:\n"
            "  • origin (0,0) at the top‐left corner\n"
            "  • x increases to the right, y increases downward\n"
            "  • bounding box format: [x0, y0, x1, y1]\n"
            "All coordinates must be normalized floats in [0.00,1.00].\n\n"
            "Text prompt (the objects to find):\n"
            '    "{instruction}"\n\n'
            "Return **only** a JSON array of objects, each with:\n"
            '  - "label": the detected object label (string)\n'
            '  - "bbox": [x0, y0, x1, y1] in normalized floats\n\n'
            "Example for a 100×200 px image:\n"
            "  A box from (10,20) to (30,120) → normalized [0.10,0.10,0.30,0.60]\n\n"
            "Expected output example:\n"
            "[\n"
            '  {{"label":"person","bbox":[0.10,0.20,0.30,0.60]}},\n'
            '  {{"label":"hardhat","bbox":[0.50,0.10,0.70,0.30]}}\n'
            "]\n\n"
            "Do NOT wrap in markdown or code fences.  No extra commentary."
        ),
        metadata={"help": "Prompt template for VLM‐based object detection"},
    )

    vlm_detection: str = field(
        default=(
            "You are an object detector.  Strictly return **only** a JSON array of objects, each with:\n"
            '  - "label": string\n'
            '  - "bbox": [x0,y0,x1,y1] (normalized floats in [0.0–1.0])\n\n'
            "Output format:\n"
            "[\n"
            '  {{"label":"person","bbox":[0.10,0.20,0.30,0.60]}},\n'
            '  {{"label":"hat","bbox":[0.50,0.10,0.70,0.30]}}\n'
            "]\n\n"
            'Instruction: "{instruction}"\n'
            "<img>"
        ),
        metadata={"help": "Prompt template for VLM-based object detection"},
    )

    vlm_detection_qwen_prompt: str = field(
        default=(
            "You are a state‑of‑the‑art vision‑language model specialized in zero‑shot object detection "
            "and visual grounding.\n\n"
            'Text prompt (objects to detect): "{instruction}"\n\n'
            "Return their locations in the form of coordinates in the format {'bbox_2d': [x1, y1, x2, y2]}.\n"
            "Include a top‑level 'count' field with the total number of objects detected in JSON format.\n\n"
            "Output format:\n"
            "{\n"
            '  "objects": [\n'
            '    {{"label":"person","bbox_2d":[341,258,397,360]}},\n'
            '    {{"label":"helmet","bbox_2d":[552,102,645,198]}}\n'
            "  ],\n"
            '  "count": 2\n'
            "}\n\n"
            "<img>"
        ),
        metadata={"help": "Prompt template for Qwen2.5‑VL based object detection"},
    )

    count_objects: str = field(
        default=(
            "You are a visual‐reasoning assistant tasked only with *counting* instances of specified objects "
            "in an image. Examine the entire image thoroughly, to ensure no instance is missed.\n\n"
            "Text prompt (objects to count):\n"
            '    "{instruction}"\n\n'
            "Return **only** a JSON object mapping each object label to its integer count.\n\n"
            "Example output:\n"
            '{{"person": 3, "hardhat": 2}}\n\n'
            "Do NOT wrap in markdown or code fences. No extra commentary."
        ),
        metadata={"help": "Prompt template for VLM‐based object counting"},
    )


# … add new templates here …
