from dataclasses import dataclass, field
from typing import Tuple, List
from core.vlm.prompt import PromptTemplates


@dataclass
class PipelineConfig:
    """All pipeline settings."""

    # ——— VLM & generation settings ———

    llm: str = field(
        default="qwen",
        metadata={"help": "LLM to use (options: 'openai', 'ollama', or 'qwen')"},
    )

    # OpenAI settings
    openai_model: str = field(default="gpt-4o", metadata={"help": "OpenAI model name"})
    temperature: float = field(
        default=0.2, metadata={"help": "Temperature for LLM generation"}
    )
    openai_concurrency: int = field(
        default=4, metadata={"help": "Concurrency for OpenAI batch calls"}
    )
    openai_batch_mode: bool = field(
        default=False,
        metadata={"help": "Use OpenAI Batch API"},
    )

    # Ollama settings
    ollama_model: str = field(
        default="qwen2.5vl:latest", metadata={"help": "Ollama model name"}
    )
    ollama_concurrency: int = field(
        default=1, metadata={"help": "Concurrency for Ollama parallel calls"}
    )

    # Qwen-VL settings
    qwen_model: str = field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        metadata={"help": "Hugging Face Qwen2.5-VL model identifier"},
    )
    qwen_concurrency: int = field(
        default=1, metadata={"help": "Concurrency for Qwen-VL parallel calls"}
    )

    # ——— Prompt templates ———

    templates: PromptTemplates = field(
        default_factory=PromptTemplates,
        metadata={"help": "VLM prompt templates & category lists"},
    )

    # ——— Grounding DINO settings ———

    groundingdino_cfg: str = field(
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        metadata={"help": "Config path for Grounding DINO"},
    )
    groundingdino_weights: str = field(
        default="GroundingDINO/groundingdino/weights/groundingdino_swint_ogc.pth",
        metadata={"help": "Weights path for Grounding DINO"},
    )
    groundingdino_hf_model: str = field(
        default="IDEA-Research/grounding-dino-tiny",
        metadata={
            "help": (
                "Hugging Face model ID for Grounding DINO HF integration\n"
                "(options: IDEA-Research/grounding-dino-tiny, IDEA-Research/grounding-dino-base)"
            )
        },
    )

    gdino_box_threshold: float = field(
        default=0.34, metadata={"help": "Box confidence threshold for Grounding DINO"}
    )
    gdino_text_threshold: float = field(
        default=0.25, metadata={"help": "Text confidence threshold for Grounding DINO"}
    )

    # ——— OWLv2 settings ———
    owlv2_model_name: str = field(
        default="google/owlv2-base-patch16-ensemble",
        metadata={
            "help": "HuggingFace checkpoint for OWLv2 zero‑shot object detection"
        },
    )
    owlv2_default_box_threshold: float = field(
        default=0.05,  # Changed from 0.10 to be more sensitive
        metadata={"help": "Box confidence threshold for OWLv2 zero-shot detection"},
    )
    owlv2_nfov_box_threshold: float = field(
        default=0.08,  # Changed from 0.15 to be more sensitive
        metadata={"help": "Box confidence threshold for OWLv2 zero-shot detection"},
    )

    # ——— SAM settings ———

    sam_version: str = field(
        default="v2", metadata={"help": "Which SAM to use: 'v1' or 'v2'"}
    )

    # SAM v1 (segment_anything)
    sam1_model_type: str = field(
        default="vit_b",
        metadata={"help": "SAM-1 ViT backbone: vit_h, vit_l, or vit_b"},
    )
    sam_checkpoint_v1: str = field(
        default="segment_anything/checkpoints/sam_vit_b_01ec64.pth",
        metadata={
            "help": "Path to SAM-v1 checkpoint (options: sam_vit_h_4b8939.pth, sam_vit_b_01ec64.pth)"
        },
    )

    # SAM 2  (sam2)
    sam2_cfg: str = field(
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        metadata={
            "help": "SAM-2 model config (options: sam2.1_hiera_s, sam2.1_hiera_b+, sam2.1_hiera_l)"
        },
    )
    sam2_checkpoint: str = field(
        default="sam2_repo/checkpoints/sam2.1_hiera_base_plus.pt",
        metadata={"help": "Path to SAM-2 checkpoint"},
    )

    sam_min_mask_iou: float = field(
        default=0.6,  # Changed from 0.85 to be more lenient
        metadata={"help": "Filter out SAM masks whose predicted quality score < this"},
    )

    # ——— Projection settings ———

    # Processing mode: "full", "cube", or "nfov"
    erp_mode: str = field(
        default="cube", metadata={"help": "Detection mode: full, cube, or nfov"}
    )

    # → Which cube faces to process (0=front,1=right,2=back,3=left,4=up,5=down)
    cube_faces: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5],  # Added face 4 (up) to catch ceiling artifacts
        metadata={"help": "Indices of cubemap faces to run detection on"},
    )

    # Cubemap settings
    face_size: int = field(
        default=512, metadata={"help": "Size (px) for each cubemap face"}
    )

    # NFoV settings
    nfov_fov_deg: List[float] = field(
        default_factory=lambda: [90.0, 90.0, 90.0, 75.0],
        metadata={"help": "FoV per pitch (deg); bottom ring smaller to overlap more"},
    )
    nfov_stride_yaw: float = field(
        default=45.0, metadata={"help": "Yaw step (deg) between successive NFoV crops"}
    )
    nfov_yaw_offset: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 22.5],
        metadata={"help": "per-pitch yaw offsets (deg)"},
    )

    # Instead of auto-generating every 30° from top to bottom,
    # pick just the three most useful pitches (middle band + a little up/down)
    nfov_pitch_angles: List[float] = field(
        default_factory=lambda: [30.0, 0.0, -30.0, -60.0],
        metadata={"help": "Exact pitch angles (deg) for NFoV crops"},
    )

    # optionally include the +180° "seam" view to catch that wrap-around boundary
    nfov_include_seam: bool = field(
        default=True,
        metadata={"help": "Whether to also sample the 180° yaw seam view"},
    )
    nfov_out_hw: Tuple[int, int] = field(
        default=(1024, 1024), metadata={"help": "Output (H,W) for each NFoV crop"}
    )

    # ——— Pipeline settings ———

    frame_rate: int = field(
        default=1, metadata={"help": "Frame rate (fps) for processing"}
    )
    skip_zero: bool = field(
        default=True, metadata={"help": "Skip DINO+SAM when VLM count is zero"}
    )
    eps_degrees: float = field(
        default=10.0,
        metadata={"help": "Angular distance threshold for spherical clustering"},
    )
    nms_iou_thresh: float = field(
        default=0.3,  # Changed from 0.4 to be more lenient
        metadata={"help": "IoU threshold for hard‐NMS"},
    )
    seam_max_w_frac: float = field(
        default=0.95,  # Changed from 0.85 to allow wider objects
        metadata={"help": "Max width fraction for seam artifact filter"},
    )
    seam_max_h_frac: float = field(
        default=0.10,  # Changed from 0.05 to allow taller objects
        metadata={"help": "Max height fraction for seam artifact filter"},
    )

    # ——— Inference settings ———

    batch_size: int = field(
        default=16,
        metadata={"help": "Frames per batch for batched inference"},
    )

    # ——— Parallelism ———

    cpu_workers: int = field(
        default=4,
        metadata={"help": "Number of CPU threads for detection"},
    )
    gpu_batch_size: int = field(
        default=6,
        metadata={"help": "Batch size per GPU pass"},
    )
