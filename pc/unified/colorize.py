"""Image colorization: DDColor + DeOldify.

DDColor predicts chroma (ab channels in Lab space) from a grayscale image.
Luminance comes from the source, so the original structure is preserved.

Available models:
- deoldify-artistic: ~250 MB ONNX, most chroma, keeps original luminance (default)
- ddcolor-full: convnext-l, ~435 MB, photorealistic
- ddcolor-artistic: ~980 MB, more artistic chroma
"""
import base64
import io
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
INPUT_SIZE = 512

COLORIZE_MODELS = {
    "ddcolor-full": {
        "path": MODEL_DIR / "ddcolor-full" / "ddcolor-512-fp16.onnx",
        "type": "ddcolor",
        "size": 512,
    },
    "ddcolor-artistic": {
        "path": MODEL_DIR / "ddcolor-artistic" / "ddcolor_artistic.onnx",
        "type": "ddcolor",
        "size": 512,
    },
    "deoldify-artistic": {
        "path": MODEL_DIR / "deoldify-artistic" / "deoldify_artistic.onnx",
        "type": "deoldify",
        "size": 256,
    },
}

_SESSIONS = {}
_DEFAULT_MODEL = "deoldify-artistic"
_ACTIVE_ID = _DEFAULT_MODEL

# Chroma boost per model. ddcolor-full needs more because the
# edge-aware smoothing flattens its noisier chroma; artistic is already richer.
SATURATION_BOOST = {
    "ddcolor-full": 3.0,
    "ddcolor-artistic": 2.2,
    "deoldify-artistic": 1.0,
}

# Chroma cast correction per model. ddcolor-full predicts a ~7 red offset in
# the 'a' channel, ddcolor-artistic already produces stronger chroma so it
# needs a gentler boost; deoldify outputs full RGB and needs none.
CAST_CORRECTION = {
    "ddcolor-full": 2.2,
    "ddcolor-artistic": 2.2,
    "deoldify-artistic": 1.0,
}

# Chroma clamp after scaling: keeps colors within valid Lab range.
AB_CLAMP = 127.0


def _ensure_nvidia_dlls():
    try:
        from unified.onnx_env import register_nvidia_dlls
        register_nvidia_dlls()
    except Exception:
        pass


def _get_session(model_id: str = _DEFAULT_MODEL):
    global _SESSIONS, _ACTIVE_ID
    if model_id in _SESSIONS:
        return _SESSIONS[model_id][0]
    spec = COLORIZE_MODELS.get(model_id)
    if not spec or not spec["path"].exists():
        logger.warning("Colorize model not found: %s", spec["path"] if spec else model_id)
        return None
    try:
        _ensure_nvidia_dlls()
        import onnxruntime as ort
        providers = ort.get_available_providers()
        session = ort.InferenceSession(str(spec["path"]), providers=providers)
        _SESSIONS[model_id] = (session, providers)
        _ACTIVE_ID = model_id
        logger.info("Colorize model loaded: %s", model_id)
        return session
    except Exception as e:
        logger.error("Colorize model load failed: %s", model_id, e)
        return None


def release() -> bool:
    global _SESSIONS, _ACTIVE_ID
    had = len(_SESSIONS) > 0
    _SESSIONS.clear()
    _ACTIVE_ID = _DEFAULT_MODEL
    return had


def is_ready() -> bool:
    return len(_SESSIONS) > 0


def _has_cv2() -> bool:
    return cv2 is not None


def _bgr_to_lab_l(bgr):
    if _has_cv2():
        lab = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2LAB)
        return lab[:, :, 0].astype(np.float32)
    # Fallback manual RGB->Lab L channel
    rgb = bgr[:, :, ::-1]
    mask = rgb <= 0.04045
    linear = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = linear @ M.T
    y = xyz[:, :, 1]
    delta = 6 / 29
    fy = np.where(y > delta ** 3, np.cbrt(y), y / (3 * delta ** 2) + 4 / 29)
    L = 116 * fy - 16
    return L.astype(np.float32)


def colorize_b64(image_b64: str, model_id: str = _DEFAULT_MODEL) -> dict:
    """Colorize a base64-encoded grayscale/B&W image.

    Supports: ddcolor-full, ddcolor-artistic, deoldify-artistic.
    Returns {"image": base64_png, "engine": model_id, "elapsed": seconds}.
    """
    t0 = time.time()
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    orig_w, orig_h = img.size

    spec = COLORIZE_MODELS.get(model_id, COLORIZE_MODELS["ddcolor-full"])
    mtype = spec["type"]
    size = spec["size"]
    session = _get_session(model_id if model_id in COLORIZE_MODELS else "ddcolor-full")
    if session is None:
        raise RuntimeError(f"Colorize model not loaded: {model_id}")

    if mtype == "ddcolor":
        # DDColor: predict ab channels, merge with L from original image.
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        L_full = _bgr_to_lab_l(img_bgr)

        img_resized = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
        L_512 = _bgr_to_lab_l(img_resized)

        lab_gray = np.zeros((size, size, 3), dtype=np.float32)
        lab_gray[:, :, 0] = L_512
        bgr_gray = cv2.cvtColor(lab_gray.astype(np.float32), cv2.COLOR_LAB2BGR).astype(np.float32)
        rgb_gray = bgr_gray[:, :, ::-1]
        inp = rgb_gray.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        outputs = session.run(None, {session.get_inputs()[0].name: inp})
        ab = outputs[0][0].transpose(1, 2, 0)  # (512, 512, 2)

        # Remove the checkpoint's global colour cast, then scale to
        # vivid but not extreme chroma.
        cast = CAST_CORRECTION.get(model_id, 0.0)
        if cast:
            for c in range(2):
                ch = ab[:, :, c]
                ab[:, :, c] = ch - cast * float(np.median(ch))
        ab_scale = np.percentile(np.abs(ab), 99.5)
        if ab_scale > 0:
            ab = ab / ab_scale * SATURATION_BOOST.get(model_id, 2.0)
        ab = np.clip(ab, -AB_CLAMP, AB_CLAMP)

        ab_full = cv2.resize(ab, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        lab_out = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
        lab_out[:, :, 0] = L_full
        # Swap a and b: model outputs (b, a) not (a, b)
        lab_out[:, :, 1] = ab_full[:, :, 1]
        lab_out[:, :, 2] = ab_full[:, :, 0]

        rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
        rgb_out = np.clip(rgb_out, 0, 1)
        out_uint8 = (rgb_out * 255).astype(np.uint8)

    elif mtype == "deoldify":
        # DeOldify: ImageNet-normalized RGB input, then facefusion-style merge
        # (preserve original L, take color from model output).
        img_resized = img.resize((size, size), Image.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr_norm = (arr - mean) / std
        inp = arr_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        outputs = session.run(None, {session.get_inputs()[0].name: inp})
        out = outputs[0][0].transpose(1, 2, 0)
        out = out * std + mean
        out = np.clip(out, 0, 1)

        # The model output has a blue bias; swap R and B to correct it.
        out_rgb = out[:, :, ::-1]
        out_img = Image.fromarray((out_rgb * 255).astype(np.uint8))
        out_img = out_img.resize((orig_w, orig_h), Image.LANCZOS)
        out_uint8 = np.array(out_img)

    out_rgb = Image.fromarray(out_uint8, mode="RGB")
    buf = io.BytesIO()
    out_rgb.save(buf, format="PNG")
    b64_out = base64.b64encode(buf.getvalue()).decode()

    elapsed = round(time.time() - t0, 2)
    logger.info("Colorized %dx%d in %.2fs using %s", orig_w, orig_h, elapsed, model_id)
    return {"image": b64_out, "elapsed": elapsed, "engine": model_id}