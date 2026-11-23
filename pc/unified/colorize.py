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

# Chroma boost per model. ddcolor-artistic already produces stronger chroma,
# so it gets a gentler boost; deoldify outputs full RGB and needs none.
# Chroma gain after cast removal. ddcolor-full needs more because the
# edge-aware smoothing flattens its noisier chroma; artistic is already richer.
SATURATION_BOOST = {
    "ddcolor-full": 3.0,
    "ddcolor-artistic": 2.2,
    "deoldify-artistic": 1.0,
}

# Both DDColor checkpoints predict a constant chroma offset for a whole image:
# ddcolor-full sits at a≈+7.5 (red cast), ddcolor-artistic at b≈+12.9 (yellow
# cast). Subtracting the per-image median recentres the scene without touching
# local colour. Values are the fraction of the median that is removed.
CAST_CORRECTION = {
    "ddcolor-full": 1.0,
    "ddcolor-artistic": 1.0,
}

# Natural photographs stay well inside ±60 in Lab a/b.
AB_CLAMP = 60.0

# Edge-aware chroma smoothing strength. The full checkpoint is noisier than
# the artistic one, which is why only the former showed fringing.
CHROMA_SMOOTH = {
    "ddcolor-full": 1,
    "ddcolor-artistic": 1,
    "deoldify-artistic": 0,
}


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 6,
                   eps: float = 120.0) -> np.ndarray:
    """Single-channel guided filter (He et al.) using box filters.

    Keeps edges from `guide` while flattening `src` inside flat regions. Used to
    clean per-pixel chroma noise without blurring colour across object borders.
    """
    if not _has_cv2():
        return src
    g = guide.astype(np.float32)
    p = src.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.blur(g, ksize)
    mean_p = cv2.blur(p, ksize)
    corr_ip = cv2.blur(g * p, ksize)
    corr_ii = cv2.blur(g * g, ksize)
    cov_ip = corr_ip - mean_i * mean_p
    var_i = corr_ii - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.blur(a, ksize)
    mean_b = cv2.blur(b, ksize)
    return mean_a * g + mean_b


def _ensure_nvidia_dlls():
    try:
        from unified.onnx_env import register_nvidia_dlls
        register_nvidia_dlls()
    except Exception as e:
        logger.debug("onnx_env register failed: %s", e)


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
        logger.error("Colorize model load failed: %s: %s", model_id, e)
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


def colorize_b64(image_b64: str, model_id: str = _DEFAULT_MODEL) -> dict:
    """Colorize a base64-encoded grayscale/B&W image.

    Supports: ddcolor-full, ddcolor-artistic, deoldify-artistic.
    Returns {"image": base64_png, "engine": model_id, "elapsed": seconds}.
    """
    t0 = time.time()
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    orig_w, orig_h = img.size

    spec = COLORIZE_MODELS.get(model_id, COLORIZE_MODELS[_DEFAULT_MODEL])
    mtype = spec["type"]
    size = spec["size"]
    session = _get_session(model_id if model_id in COLORIZE_MODELS else _DEFAULT_MODEL)
    if session is None:
        raise RuntimeError(f"Colorize model not loaded: {model_id}")

    if mtype == "ddcolor":
        # Facefusion approach: convert to grayscale, L channel, predict ab
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        gray_rgb = (gray_rgb / 255.0).astype(np.float32)
        # L channel from grayscale
        lab = cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2LAB)
        L_small = lab[:, :, 0:1]  # (H, W, 1)
        # Concatenate L + 0 + 0 -> Lab with zero chroma
        lab_zero = np.concatenate([L_small, np.zeros_like(L_small), np.zeros_like(L_small)], axis=-1)
        rgb_input = cv2.cvtColor(lab_zero.astype(np.float32), cv2.COLOR_LAB2RGB)
        rgb_input = cv2.resize(rgb_input, (size, size), interpolation=cv2.INTER_AREA)
        # Luminance at model resolution: used as the edge guide for chroma.
        L_guide = cv2.cvtColor(rgb_input.astype(np.float32), cv2.COLOR_RGB2LAB)[:, :, 0]
        inp = rgb_input.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        outputs = session.run(None, {session.get_inputs()[0].name: inp})
        ab = outputs[0][0].transpose(1, 2, 0)  # (size, size, 2)

        # Remove the checkpoint's global colour cast, then scale to a natural
        # photographic chroma range. Values are in CIE Lab a/b units.
        cast = CAST_CORRECTION.get(model_id, 0.0)
        if cast:
            for c in range(2):
                ab[:, :, c] -= cast * float(np.median(ab[:, :, c]))
        ab = ab * SATURATION_BOOST.get(model_id, 2.0)
        ab = np.clip(ab, -AB_CLAMP, AB_CLAMP)

        # Full-res L from original image
        img_bgr_f = (img_bgr / 255.0).astype(np.float32)
        L_full = cv2.cvtColor(img_bgr_f, cv2.COLOR_BGR2LAB)[:, :, 0:1]

        # Edge-aware smoothing of the predicted chroma. The checkpoint emits
        # per-pixel chroma noise; amplifying it and upscaling with plain
        # bilinear produced fringes along high-contrast edges (the red/blue
        # "3D glasses" look). A guided filter keyed on the model-resolution
        # luminance removes the speckle and keeps chroma aligned to edges.
        if CHROMA_SMOOTH.get(model_id, 0):
            for c in range(2):
                ab[:, :, c] = _guided_filter(L_guide, ab[:, :, c],
                                             radius=6, eps=120.0)

        # Resize ab to original size
        ab_full = cv2.resize(ab, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Merge L + ab -> Lab -> RGB (use RGB directly, not BGR)
        lab_out = np.concatenate([L_full, ab_full], axis=-1).astype(np.float32)
        rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
        rgb_out = np.clip(rgb_out, 0, 1)
        # Chroma is already scaled in the a/b domain; no extra HSV boost, which
        # used to wash the hue and made the result look posterised.
        out_uint8 = (np.clip(rgb_out, 0, 1) * 255).astype(np.uint8)

    elif mtype == "deoldify":
        # DeOldify (FaceFusion reference pipeline):
        #   input  = grayscale RGB, 0-255 float, no normalization
        #   output = RGB 0-255
        # The model's chroma is merged into the ORIGINAL luminance so the
        # result keeps the source structure instead of becoming a redraw.
        src_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        resized = cv2.resize(gray_rgb, (size, size))
        inp = resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        outputs = session.run(None, {session.get_inputs()[0].name: inp})
        out = outputs[0][0].transpose(1, 2, 0)          # RGB, 0-255
        out_resized = cv2.resize(out, (orig_w, orig_h))

        # Keep the original L (blue channel is a faithful luminance proxy),
        # take only chroma (a, b) from the colorized output.
        blue_ch, _, _ = cv2.split(src_bgr)
        color = cv2.cvtColor(out_resized.astype(np.uint8), cv2.COLOR_BGR2RGB)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
        _, a_ch, b_ch = cv2.split(color)
        merged = cv2.cvtColor(cv2.merge((blue_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        out_img = Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
        out_uint8 = np.array(out_img)

    out_rgb = Image.fromarray(out_uint8, mode="RGB")
    buf = io.BytesIO()
    out_rgb.save(buf, format="PNG")
    b64_out = base64.b64encode(buf.getvalue()).decode()

    elapsed = round(time.time() - t0, 2)
    logger.info("Colorized %dx%d in %.2fs using %s", orig_w, orig_h, elapsed, model_id)
    return {"image": b64_out, "elapsed": elapsed, "engine": model_id}
