"""Colorization via DDColor (ONNX, ICCV 2023).

DDColor predicts chroma (ab channels in Lab space) from a grayscale image.
Luminance comes from the source, so the original structure is preserved.

Model: ddcolor-tiny (edgetools/ddcolor, ~130 MB, fp16, 512x512, Apache-2.0)
Input:  (1, 3, 512, 512) float32 RGB grayscale (Lab L rebuilt as RGB)
Output: (1, 2, 512, 512) float32 ab channels
"""
import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

MODEL_ID = "ddcolor-tiny"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "ddcolor-tiny"
MODEL_FILE = MODEL_DIR / "ddcolor-tiny-fp16.onnx"
INPUT_SIZE = 512

_SESSION = None
_PROVIDER = None


def _ensure_nvidia_dlls():
    """Register NVIDIA DLL directories so onnxruntime-gpu can find cuDNN."""
    try:
        from unified.onnx_env import register_nvidia_dlls
        register_nvidia_dlls()
    except Exception as e:
        logger.debug("onnx_env register failed: %s", e)


def _get_session() -> Optional[object]:
    global _SESSION, _PROVIDER
    if _SESSION is not None:
        return _SESSION
    if not MODEL_FILE.exists():
        logger.warning("DDColor model not found: %s", MODEL_FILE)
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
        logger.info("DDColor loaded: provider=%s", _PROVIDER)
        return _SESSION
    except Exception as e:
        logger.error("DDColor load failed: %s", e)
        return None


def release():
    global _SESSION, _PROVIDER
    _SESSION = None
    _PROVIDER = None


def is_ready() -> bool:
    return _SESSION is not None


def _has_cv2() -> bool:
    return cv2 is not None


def _bgr_to_lab_l(bgr: np.ndarray) -> np.ndarray:
    if _has_cv2():
        lab = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2LAB)
        return lab[:, :, 0].astype(np.float32)
    rgb = bgr[:, :, ::-1]
    mask = rgb <= 0.04045
    linear = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = linear @ M.T
    y = xyz[:, :, 1] / 1.0
    delta = 6 / 29
    fy = np.where(y > delta ** 3, np.cbrt(y), y / (3 * delta ** 2) + 4 / 29)
    L = 116 * fy - 16
    return L.astype(np.float32)


def _lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    if _has_cv2():
        return cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200
    delta = 6 / 29
    def finv(t):
        return np.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4 / 29))
    X = 0.95047 * finv(x)
    Y = 1.0 * finv(y)
    Z = 1.08883 * finv(z)
    xyz = np.stack([X, Y, Z], axis=-1)
    M_inv = np.array([[3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660, 1.8760108, 0.0415560],
                      [0.0556434, -0.2040259, 1.0572252]], dtype=np.float32)
    rgb = xyz @ M_inv.T
    mask = rgb > 0.0031308
    rgb = np.where(mask, 1.055 * np.power(np.maximum(rgb, 0), 1 / 2.4) - 0.055,
                   12.92 * rgb)
    rgb = np.clip(rgb, 0, 1)
    return rgb[:, :, ::-1].astype(np.float32)


def _rgb_to_lab_l(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-1 float) to Lab L channel (0-100). Uses cv2 if available."""
    try:
        import cv2
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        return lab[:, :, 0]
    except ImportError:
        # Fallback: manual RGB->XYZ->Lab
        # sRGB -> linear
        mask = rgb <= 0.04045
        linear = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        # XYZ matrix (D65)
        M = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
        xyz = linear @ M.T
        # Normalize by D65 white
        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        x, y, z = xyz[:, :, 0] / Xn, xyz[:, :, 1] / Yn, xyz[:, :, 2] / Zn
        def f(t):
            delta = 6 / 29
            return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4 / 29)
        L = 116 * f(y) - 16
        return L


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert Lab (L 0-100, a,b ~[-127,127]) to RGB (0-1 float)."""
    try:
        import cv2
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except ImportError:
        # Manual Lab->XYZ->RGB
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        y = (L + 16) / 116
        x = a / 500 + y
        z = y - b / 200
        delta = 6 / 29
        def finv(t):
            return np.where(t > delta, t**3, 3 * delta**2 * (t - 4 / 29))
        X = 0.95047 * finv(x)
        Y = 1.0 * finv(y)
        Z = 1.08883 * finv(z)
        xyz = np.stack([X, Y, Z], axis=-1)
        M_inv = np.array([[3.2404542, -1.5371385, -0.4985314],
                          [-0.9692660, 1.8760108, 0.0415560],
                          [0.0556434, -0.2040259, 1.0572252]], dtype=np.float32)
        rgb = xyz @ M_inv.T
        # sRGB gamma
        mask = rgb > 0.0031308
        rgb = np.where(mask, 1.055 * np.power(np.maximum(rgb, 0), 1 / 2.4) - 0.055,
                       12.92 * rgb)
        return np.clip(rgb, 0, 1)


def colorize_b64(image_b64: str) -> dict:
    """Colorize a base64-encoded grayscale/B&W image using DDColor.

    Returns {"image": base64_png, "engine": "ddcolor", "elapsed": seconds}.
    """
    import base64
    from PIL import Image

    t0 = time.time()
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    orig_w, orig_h = img.size

    # Convert to numpy BGR float [0,1] (OpenCV convention — model trained on BGR)
    img_bgr = np.array(img, dtype=np.float32)[:, :, ::-1] / 255.0

    # Compute full-resolution L channel (BGR -> Lab)
    L_full = _bgr_to_lab_l(img_bgr)  # (H, W)

    # Resize to 512x512
    img_512 = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA) if _has_cv2() else np.array(img.resize((INPUT_SIZE, INPUT_SIZE)), dtype=np.float32)[:, :, ::-1] / 255.0

    # Compute L for the 512 version
    L_512 = _bgr_to_lab_l(img_512)  # (512, 512)

    # Rebuild grayscale BGR from L (a=b=0, Lab->BGR)
    lab_gray = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    lab_gray[:, :, 0] = L_512
    # a and b stay 0
    bgr_gray = _lab_to_bgr(lab_gray)  # (512, 512, 3) float32 [0,1]

    # Input: (1, 3, 512, 512) float32 NCHW (RGB order for the model)
    # DDColor was trained on BGR images converted to RGB for the model input
    # The model expects RGB input (OpenCV BGR -> RGB for feeding)
    rgb_gray = bgr_gray[:, :, ::-1]
    inp = rgb_gray.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    # Run inference
    session = _get_session()
    if session is None:
        raise RuntimeError("DDColor model not loaded")

    outputs = session.run(None, {session.get_inputs()[0].name: inp})
    ab = outputs[0]  # (1, 2, 512, 512) float32
    ab = ab[0].transpose(1, 2, 0)  # (512, 512, 2)

    # Upscale ab to original size
    ab_full = cv2.resize(ab, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Merge with full-res L
    lab_out = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
    lab_out[:, :, 0] = L_full
    lab_out[:, :, 1:] = ab_full

    # Convert Lab -> RGB (reference uses COLOR_LAB2RGB for output)
    rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
    rgb_out = np.clip(rgb_out, 0, 1)

    # Convert to uint8 and encode as PNG
    out_uint8 = (rgb_out * 255).astype(np.uint8)
    out_rgb = Image.fromarray(out_uint8, mode="RGB")
    buf = io.BytesIO()
    out_rgb.save(buf, format="PNG")
    b64_out = base64.b64encode(buf.getvalue()).decode()

    elapsed = round(time.time() - t0, 2)
    logger.info("DDColor colorized %dx%d in %.2fs", orig_w, orig_h, elapsed)
    return {"image": b64_out, "elapsed": elapsed, "engine": "ddcolor-tiny"}
