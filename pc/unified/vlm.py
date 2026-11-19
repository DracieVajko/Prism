"""VLM caption/chat - Qwen3-VL-2B via the shared OCR engine (PC).

Qwen3-VL-2B replaced SmolVLM-500M on the PC: it won the image-description
comparison and is already used for OCR, so one resident model serves
`/v1/images/caption`, vision chat, and `/v1/images/ocr?task=describe`.

Weights: models/qwen3-vl-2b/ (`download_models.py qwen3-vl-2b`, ~4.3 GB,
Apache-2.0). Without weights: EngineMissing -> the endpoint returns 503
will-not-load (not 500). Android keeps its own SmolVLM2 runtime.
"""
import base64
import io
import logging
import time
from pathlib import Path

from PIL import Image

logger = logging.getLogger("unified.vlm")

_HERE = Path(__file__).resolve().parent.parent
MODEL_ID = "qwen3-vl-2b"
_DEFAULT_DIRS = (_HERE / "models" / MODEL_ID,)


class EngineMissing(Exception):
    pass


def find_vlm_dir(dirs=None):
    for d in (dirs if dirs is not None else _DEFAULT_DIRS):
        d = Path(d)
        if ((d / "model.safetensors").is_file()
                or (d / "model.safetensors.index.json").is_file()
                or (d / "pytorch_model.bin").is_file()):
            return d
    return None


def release_vlm() -> bool:
    """Drop the shared Qwen3-VL engine so diffusion/OCR can take the VRAM."""
    try:
        from unified.ocr_engine import reset_ocr_engine
        reset_ocr_engine(MODEL_ID)
    except Exception as exc:
        logger.debug("release_vlm failed: %s", exc)
    return True


def caption_image_b64(image_b64: str, prompt: str = "Describe the image in detail.",
                      max_tokens: int = 256, search_dirs=None) -> dict:
    try:
        raw = base64.b64decode(image_b64)
    except Exception as e:
        raise ValueError(f"bad base64 image: {e}")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"bad image bytes: {e}")
    model_dir = find_vlm_dir(search_dirs)
    if model_dir is None:
        raise EngineMissing(
            f"VLM weights missing ({MODEL_ID}); run "
            f"python download_models.py {MODEL_ID}")
    t0 = time.time()
    try:
        from unified.ocr_engine import get_ocr_engine
        engine = get_ocr_engine(MODEL_ID)
        if not engine.init():
            raise EngineMissing(f"VLM model failed to load ({MODEL_ID})")
        text = engine.generate_pil(
            img, prompt or "Describe the image.",
            max_new_tokens=max(16, min(512, max_tokens)),
        )
    except EngineMissing:
        raise
    except Exception as e:
        raise EngineMissing(f"VLM inference failed: {str(e)[:180]}") from e
    return {"text": text, "elapsed": round(time.time() - t0, 2),
            "engine": MODEL_ID}
