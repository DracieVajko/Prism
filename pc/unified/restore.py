"""Image restoration via InstructPix2Pix (diffusers, SD 1.5 based).

InstructPix2Pix is an instruction-based image editing model that can
restore, repair, and reconstruct images following natural language prompts.
Unlike DDColor (colorization only) or img2img (style transfer), it
understands the full image context and can perform targeted restoration.

Model: timbrooks/instruct-pix2pix (~5.8 GB, SD 1.5 based, MIT license)
"""
import base64
import io
import logging
import time
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

MODEL_ID = "instruct-pix2pix"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "instruct-pix2pix"

_PIPELINE = None


def _ensure_pipeline() -> object | None:
    """Load InstructPix2Pix on first use.

    torch and diffusers are imported only after the weights are confirmed to
    exist, so importing this module works on machines without the ML stack and
    a missing model yields a clean unavailable state instead of an ImportError.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    if not (MODEL_DIR / "model_index.json").exists():
        logger.warning("InstructPix2Pix model not found: %s", MODEL_DIR)
        return None
    try:
        import torch
        from diffusers import StableDiffusionInstructPix2PixPipeline
    except ImportError as e:
        logger.warning("InstructPix2Pix dependencies unavailable: %s", e)
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "safety_checker": None,
            "use_safetensors": True,
        }
        if device == "cuda":
            kwargs["variant"] = "fp16"
        _PIPELINE = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            str(MODEL_DIR), **kwargs,
        ).to(device)
        logger.info("InstructPix2Pix loaded on %s", device.upper())
        return _PIPELINE
    except Exception as e:
        logger.error("InstructPix2Pix load failed: %s", e)
        return None


def release() -> bool:
    global _PIPELINE
    had = _PIPELINE is not None
    _PIPELINE = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return had


def is_ready() -> bool:
    return _PIPELINE is not None


def restore_b64(image_b64: str, prompt: str = "restore and colorize this old photo, remove damage, enhance clarity, natural daylight colors, preserve original faces and clothing", strength: float = 0.5, steps: int = 20, guidance: float = 7.5, image_guidance: float = 1.5) -> dict:
    """Restore an image using InstructPix2Pix.

    Returns {"image": base64_png, "elapsed": seconds, "engine": "instruct-pix2pix"}.
    """
    t0 = time.time()
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    orig_w, orig_h = img.size

    pipe = _ensure_pipeline()
    if pipe is None:
        raise RuntimeError("InstructPix2Pix model not loaded")

    # Resize to 512x512 for the model (SD 1.5 native resolution)
    img_resized = img.resize((512, 512), Image.LANCZOS)

    result = pipe(
        prompt,
        image=img_resized,
        num_inference_steps=steps,
        guidance_scale=guidance,
        image_guidance_scale=image_guidance,
    )
    out = result.images[0]

    # Resize back to original dimensions
    if out.size != (orig_w, orig_h):
        out = out.resize((orig_w, orig_h), Image.LANCZOS)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    b64_out = base64.b64encode(buf.getvalue()).decode()

    elapsed = round(time.time() - t0, 2)
    logger.info("InstructPix2Pix restored %dx%d in %.2fs", orig_w, orig_h, elapsed)
    return {"image": b64_out, "elapsed": elapsed, "engine": "instruct-pix2pix"}
