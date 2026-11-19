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
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_ID = "flux-kontext-dev"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "flux-kontext-dev"

_PIPELINE = None


def _ensure_pipeline() -> Optional[object]:
    global _PIPELINE
    had = _PIPELINE is not None
    if _PIPELINE is not None:
        return _PIPELINE
    if not (MODEL_DIR / "model_index.json").exists():
        logger.warning("FLUX.1-Kontext-dev model not found: %s", MODEL_DIR)
        return None
    try:
        from diffusers import FluxKontextPipeline
        _PIPELINE = FluxKontextPipeline.from_pretrained(
            str(MODEL_DIR), torch_dtype=torch.bfloat16
        )
        # Enable CPU offload for large models on limited VRAM
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
