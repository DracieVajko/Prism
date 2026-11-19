"""Super-resolution / restoration via SwinIR (ONNX, ICCV 2021).

SwinIR real-world super-resolution (4x): handles compression artifacts,
sensor noise, mild blur. Input 64x64 RGB [0,1], output 256x256 RGB [0,1].
For larger images, tiling is used (64px tiles, 8px overlap).
"""
import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_ID = "swinir-realsr-x4"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "swinir-realsr-x4"
MODEL_FILE = MODEL_DIR / "swinir_realsr_x4.onnx"
TILE_SIZE = 64
OVERLAP = 8
# Input side cap; 1024 px gives a 4096 px output. The 64 px tile keeps VRAM
# flat, so only runtime grows. Override with PRISM_SWINIR_MAX_SIDE.
MAX_SIDE = int(os.environ.get("PRISM_SWINIR_MAX_SIDE", "1024"))

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
    if _SESSION is not None:
        return _SESSION
    if not MODEL_FILE.exists():
        logger.warning("SwinIR model not found: %s", MODEL_FILE)
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
        logger.info("SwinIR loaded: provider=%s", _PROVIDER)
        return _SESSION
    except Exception as e:
        logger.error("SwinIR load failed: %s", e)
        return None


def release():
    global _SESSION, _PROVIDER
    _SESSION = None
    _PROVIDER = None


def is_ready() -> bool:
    return _SESSION is not None


def _tile_upscale(img_arr: np.ndarray, session) -> np.ndarray:
    """Upscale an image array (H, W, 3) float32 [0,1] using tiled inference."""
    h, w = img_arr.shape[:2]
    scale = 4
    out_h, out_w = h * scale, w * scale
    result = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 3), dtype=np.float32)

    # Tile positions
    step = TILE_SIZE - OVERLAP
    tiles_y = list(range(0, h - TILE_SIZE + 1, step))
    tiles_x = list(range(0, w - TILE_SIZE + 1, step))
    if tiles_y and tiles_y[-1] + TILE_SIZE < h:
        tiles_y.append(h - TILE_SIZE)
    if tiles_x and tiles_x[-1] + TILE_SIZE < w:
        tiles_x.append(w - TILE_SIZE)
    if not tiles_y:
        tiles_y = [0]
    if not tiles_x:
        tiles_x = [0]

    for ty in tiles_y:
        for tx in tiles_x:
            tile = img_arr[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]
            # Pad if needed
            pad_h = TILE_SIZE - tile.shape[0]
            pad_w = TILE_SIZE - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            inp = tile.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            out_tile = session.run(None, {session.get_inputs()[0].name: inp})[0]
            out_tile = np.clip(out_tile[0].transpose(1, 2, 0), 0, 1)
            # Remove padding
            out_h_tile = TILE_SIZE * scale - pad_h * scale
            out_w_tile = TILE_SIZE * scale - pad_w * scale
            out_tile = out_tile[:out_h_tile, :out_w_tile]

            oy, ox = ty * scale, tx * scale
            result[oy:oy + out_tile.shape[0], ox:ox + out_tile.shape[1]] += out_tile
            weight[oy:oy + out_tile.shape[0], ox:ox + out_tile.shape[1]] += 1

    # Average overlapping regions
    mask = weight > 0
    result[mask] /= weight[mask]
    return result


def upscale_b64(image_b64: str) -> dict:
    """Upscale/restore an image using SwinIR real-SR x4.

    Returns {"image": base64_png, "elapsed": seconds, "engine": "swinir-realsr-x4", "scale": 4}.
    """
    t0 = time.time()
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    orig_w, orig_h = img.size

    # Cap input size to avoid OOM on large images
    if max(orig_w, orig_h) > MAX_SIDE:
        scale = MAX_SIDE / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        orig_w, orig_h = img.size
        logger.info("SwinIR input resized to %dx%d (cap %dpx)", orig_w, orig_h, MAX_SIDE)

    arr = np.array(img, dtype=np.float32) / 255.0

    session = _get_session()
    if session is None:
        raise RuntimeError("SwinIR model not loaded")

    out = _tile_upscale(arr, session)
    out = np.clip(out, 0, 1)
    out_uint8 = (out * 255).astype(np.uint8)
    out_img = Image.fromarray(out_uint8, mode="RGB")

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    b64_out = base64.b64encode(buf.getvalue()).decode()

    elapsed = round(time.time() - t0, 2)
    logger.info("SwinIR upscaled %dx%d -> %dx%d in %.2fs", orig_w, orig_h, out_img.size[0], out_img.size[1], elapsed)
    return {"image": b64_out, "elapsed": elapsed, "engine": "swinir-realsr-x4", "scale": 4}
