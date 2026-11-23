"""Image restoration via FLUX.1-Kontext-dev (12B rectified flow transformer).

FLUX.1 Kontext is a state-of-the-art image editing/restoration model that
understands the full image context and can reconstruct damaged areas,
colorize, enhance clarity, and preserve original structure.

Model: black-forest-labs/FLUX.1-Kontext-dev (24 GB, bfloat16, non-commercial)
       or unsloth/FLUX.1-Kontext-dev (ungated mirror)
Input: image + text instruction
Output: restored/edited image

Requirements:
- 12B parameter model, ~24 GB VRAM in FP16
- With CPU offload: ~7.5 GB VRAM + 24 GB RAM, ~3-5 min per image
- Requires diffusers with FluxKontextPipeline support
"""
import base64
import io
import logging
import time
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

MODEL_ID = "flux-kontext-dev"
# Full bf16 snapshot first, then the FP8 build. Both are FluxKontextPipeline;
# FP8 is ~21.8 GB on disk and loads on machines where bf16 (~31.8 GB) will not.
MODEL_DIRS = [
    Path(__file__).resolve().parent.parent / "models" / "flux-kontext-dev",
    Path(__file__).resolve().parent.parent / "models" / "flux-kontext-dev-fp8",
]


def _model_dir() -> Path:
    for d in MODEL_DIRS:
        if (d / "model_index.json").is_file():
            return d
    return MODEL_DIRS[0]


MODEL_DIR = _model_dir()

_PIPELINE = None


def _ensure_pipeline() -> object | None:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    model_dir = _model_dir()
    if not (model_dir / "model_index.json").exists():
        logger.warning("FLUX.1-Kontext-dev model not found: %s", model_dir)
        return None
    try:
        import torch
        from diffusers import FluxKontextPipeline
        logger.info("FLUX.1-Kontext-dev loading from %s", model_dir)
        _PIPELINE = FluxKontextPipeline.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )
        # Sequential offload streams layer by layer: required on 32 GB RAM
        # hosts where the whole pipeline does not fit at once.
        try:
            _PIPELINE.enable_sequential_cpu_offload()
            logger.info("FLUX.1-Kontext-dev loaded with sequential CPU offload")
        except Exception:
            _PIPELINE.enable_model_cpu_offload()
            logger.info("FLUX.1-Kontext-dev loaded with CPU offload")
        return _PIPELINE
    except Exception as e:
        logger.error("FLUX.1-Kontext-dev load failed: %s", e)
        return None


def release() -> bool:
    global _PIPELINE
    had = _PIPELINE is not None
    _PIPELINE = None
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return had


def is_ready() -> bool:
    return _PIPELINE is not None


def restore_b64(image_b64: str, prompt: str = "restore and colorize this old photo, remove all damage including stains, cracks, and fading. Enhance the clarity and resolution, and add subtle, historically accurate natural daylight colors while strictly keeping the original faces, clothing, and background authentic.", strength: float = 2.5, steps: int = 28) -> dict:
    """Restore an image using FLUX.1-Kontext-dev.

    Returns {"image": base64_png, "elapsed": seconds, "engine": "flux-kontext-dev"}.
    """
    t0 = time.time()
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    orig_w, orig_h = img.size

    pipe = _ensure_pipeline()
    if pipe is None:
        raise RuntimeError("FLUX.1-Kontext-dev model not loaded")

    # Resize to a manageable size for the model
    max_side = 1024
    if max(orig_w, orig_h) > max_side:
        scale = max_side / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info("FLUX input resized to %dx%d (cap %dpx)", new_w, new_h, max_side)

    result = pipe(
        image=img,
        prompt=prompt,
        guidance_scale=strength,
        num_inference_steps=steps,
    )
    out = result.images[0]

    # Resize back to original dimensions if needed
    if out.size != (orig_w, orig_h):
        out = out.resize((orig_w, orig_h), Image.LANCZOS)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    b64_out = base64.b64encode(buf.getvalue()).decode()

    elapsed = round(time.time() - t0, 2)
    logger.info("FLUX.1-Kontext restored %dx%d in %.2fs", orig_w, orig_h, elapsed)
    return {"image": b64_out, "elapsed": elapsed, "engine": "flux-kontext-dev"}
