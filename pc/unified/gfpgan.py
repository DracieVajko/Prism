"""Face restoration via GFPGAN (ONNX, CVPR 2021).

GFPGAN restores degraded/blurred/damaged faces using a generative facial prior.
Input:  (1, 3, 512, 512) float32 RGB in [-1, 1] (FFHQ-aligned face crop)
Output: (1, 3, 512, 512) float32 RGB in [-1, 1]
"""
import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_ID = "gfpgan-v1.4"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gfpgan-v1.4"
MODEL_FILE = MODEL_DIR / "gfpgan14-fp16.onnx"

_SESSION = None
_PROVIDER = None


def _ensure_nvidia_dlls():
    try:
        from unified.onnx_env import register_nvidia_dlls
        register_nvidia_dlls()
    except Exception as e:
        logger.debug("onnx_env register failed: %s", e)


def _get_session() -> Optional[object]:
    global _SESSION, _PROVIDER
    had = _SESSION is not None
    if _SESSION is not None:
        return _SESSION
    if not MODEL_FILE.exists():
        logger.warning("GFPGAN model not found: %s", MODEL_FILE)
        return None
    try:
        _ensure_nvidia_dlls()
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            _PROVIDER = "cuda"
        elif "DmlExecutionProvider" in providers:
            _PROVIDER = "directml"
        else:
            _PROVIDER = "cpu"
        _SESSION = ort.InferenceSession(str(MODEL_FILE), providers=providers)
        logger.info("GFPGAN loaded: provider=%s", _PROVIDER)
        return _SESSION
    except Exception as e:
        logger.error("GFPGAN load failed: %s", e)
        return None


def release() -> bool:
    global _SESSION, _PROVIDER
    had = _SESSION is not None
    _SESSION = None
    _PROVIDER = None
    return had


def is_ready() -> bool:
    return _SESSION is not None


def restore_face_b64(image_b64: str) -> dict:
    """Restore a face crop using GFPGAN.

    Input image should be a face crop (ideally FFHQ-aligned, 512x512).
    Returns {"image": base64_png, "elapsed": seconds, "engine": "gfpgan-v1.4"}.
    """
    t0 = time.time()
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    orig_w, orig_h = img.size

    # Resize to 512x512
    img_512 = img.resize((512, 512), Image.LANCZOS)

    # Convert to [-1, 1] float32
    arr = (np.array(img_512, dtype=np.float32) / 127.5) - 1.0
    inp = arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    session = _get_session()
    if session is None:
        raise RuntimeError("GFPGAN model not loaded")

    out = session.run(None, {session.get_inputs()[0].name: inp})[0]
    out = np.clip(out[0].transpose(1, 2, 0), -1, 1)
    out = ((out + 1) / 2 * 255).astype(np.uint8)

    # Resize back to original if different
    out_img = Image.fromarray(out, mode="RGB")
    if out_img.size != (orig_w, orig_h):
        out_img = out_img.resize((orig_w, orig_h), Image.LANCZOS)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    b64_out = base64.b64encode(buf.getvalue()).decode()

    elapsed = round(time.time() - t0, 2)
    logger.info("GFPGAN restored face %dx%d in %.2fs", orig_w, orig_h, elapsed)
    return {"image": b64_out, "elapsed": elapsed, "engine": "gfpgan-v1.4"}
