"""Generative upscale refine via Z-Image Turbo img2img (Apache-2.0).

Z-Image Turbo is a 6B single-stream DiT with a Qwen3-4B text encoder and the
FLUX VAE. It is *not* a replacement for Real-ESRGAN: the caller first runs the
deterministic x4 pass, then this module re-samples the result at a low denoise
strength to rebuild texture and small detail.

Memory strategy on a 6 GB card:
    the bf16 transformer alone is ~12.3 GB, so a whole-module CPU offload would
    still OOM. Below 9 GB VRAM the pipeline uses `enable_sequential_cpu_offload`
    (layer by layer, slow but fits). Larger cards get `enable_model_cpu_offload`.

Large images are refined in tiles with a feathered overlap, because a 2048 px
pass does not fit 6 GB even with offload.

Weights: pc/models/z-image-turbo (`download_models.py --preset zimage`).
"""
import base64
import io
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Resolution order: the full bf16 snapshot first, then the FP8 build. Both
# load through the same ZImageImg2ImgPipeline; only the stored dtype differs.
MODEL_DIRS = [
    Path(__file__).resolve().parent.parent / "models" / "z-image-turbo",
    Path(__file__).resolve().parent.parent / "models" / "z-image-turbo-fp8",
]


def _model_dir() -> Path:
    """First complete snapshot (bf16 or FP8)."""
    for d in MODEL_DIRS:
        if (d / "model_index.json").is_file():
            tdir = d / "transformer"
            if tdir.is_dir() and sum(
                    f.stat().st_size for f in tdir.glob("*.safetensors")) > 4 * 1024 ** 3:
                return d
    return MODEL_DIRS[0]


MODEL_DIR = _model_dir()
DEFAULT_PROMPT = "high quality, sharp natural detail, clean edges"
DEFAULT_STRENGTH = 0.33
DEFAULT_STEPS = 9
TILE = int(os.environ.get("PRISM_ZIMAGE_TILE", "1024"))
OVERLAP = int(os.environ.get("PRISM_ZIMAGE_OVERLAP", "96"))
MAX_TILES = int(os.environ.get("PRISM_ZIMAGE_MAX_TILES", "9"))

_pipe = None
_lock = threading.Lock()


def is_installed() -> bool:
    """Complete enough to load: index + at least most of the transformer.

    The size guard matters during a download: model_index.json and the
    transformer folder appear long before the 12 GB of weights are complete.
    """
    d = _model_dir()
    if not (d / "model_index.json").is_file():
        return False
    tdir = d / "transformer"
    if not tdir.is_dir():
        return False
    total = sum(f.stat().st_size for f in tdir.glob("*.safetensors"))
    # bf16 transformer is ~12.3 GB, the FP8 build ~6.2 GB.
    return total > 4 * 1024 ** 3


def _load():
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Z-Image needs a CUDA GPU; this build has no CUDA device")
    from diffusers import ZImageImg2ImgPipeline

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    model_dir = _model_dir()
    logger.info("Z-Image loading from %s", model_dir)
    pipe = ZImageImg2ImgPipeline.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if vram_gb < 9:
        # The transformer is larger than the card; stream it layer by layer.
        pipe.enable_sequential_cpu_offload()
        logger.info("Z-Image: sequential CPU offload (%.1f GB VRAM)", vram_gb)
    else:
        pipe.enable_model_cpu_offload()
        logger.info("Z-Image: model CPU offload (%.1f GB VRAM)", vram_gb)
    for name in ("enable_vae_slicing", "enable_vae_tiling"):
        fn = getattr(pipe, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
    vae = getattr(pipe, "vae", None)
    for name in ("enable_tiling", "enable_slicing"):
        fn = getattr(vae, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    _pipe = pipe
    return pipe


def release() -> bool:
    global _pipe
    had = _pipe is not None
    _pipe = None
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return had


def _pad_to_multiple(img: Image.Image, m: int = 16) -> Image.Image:
    w, h = img.size
    nw, nh = (w + m - 1) // m * m, (h + m - 1) // m * m
    if (nw, nh) == (w, h):
        return img
    out = Image.new("RGB", (nw, nh))
    out.paste(img, (0, 0))
    return out


def _positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = tile - overlap
    pos = list(range(0, length - tile + 1, step))
    if pos[-1] + tile < length:
        pos.append(length - tile)
    return pos


def _ramp(n: int, overlap: int, left_edge: bool, right_edge: bool):
    w = np.ones(n, dtype=np.float32)
    overlap = min(overlap, max(0, n // 2))
    if not left_edge and overlap > 0:
        w[:overlap] = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    if not right_edge and overlap > 0:
        w[-overlap:] = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
    return w


def _run(pipe, img: Image.Image, prompt: str, strength: float,
         steps: int, seed: int) -> Image.Image:
    import torch
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    result = pipe(
        prompt=prompt,
        image=img,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=0.0,
        generator=gen,
    )
    return result.images[0].convert("RGB")


def refine_b64(image_b64: str, prompt: str = "", strength: float = DEFAULT_STRENGTH,
               steps: int = DEFAULT_STEPS, seed: int = 0) -> dict:
    """Refine an already-upscaled image. Returns the same dict shape as upscale."""
    if not is_installed():
        raise FileNotFoundError(
            f"Z-Image Turbo not installed in {MODEL_DIR}; run "
            "`python download_models.py --preset zimage`")
    try:
        raw = base64.b64decode(image_b64)
    except Exception as e:
        raise ValueError(f"bad base64 image: {e}")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"bad image bytes: {e}")

    prompt = (prompt or "").strip() or DEFAULT_PROMPT
    strength = min(max(float(strength), 0.05), 0.9)
    steps = max(1, int(steps))

    t0 = time.time()
    with _lock:
        pipe = _load()
        if max(img.size) <= TILE:
            out = _run(pipe, _pad_to_multiple(img), prompt, strength, steps, seed)
            out = out.crop((0, 0, img.width, img.height))
        else:
            out = _refine_tiled(pipe, img, prompt, strength, steps, seed)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "elapsed": round(time.time() - t0, 2),
        "engine": "zimage-turbo",
        "strength": strength,
        "steps": steps,
        "size": list(out.size),
    }


def _refine_tiled(pipe, img: Image.Image, prompt: str, strength: float,
                  steps: int, seed: int) -> Image.Image:
    w, h = img.size
    acc = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    xs = _positions(w, TILE, OVERLAP)
    ys = _positions(h, TILE, OVERLAP)
    if len(xs) * len(ys) > MAX_TILES:
        raise ValueError(
            f"generative refine would need {len(xs) * len(ys)} tiles for "
            f"{w}x{h}; use a smaller scale (PRISM_ZIMAGE_MAX_TILES="
            f"{MAX_TILES}, tile={TILE})")
    logger.info("Z-Image tiled refine %dx%d in %dx%d tiles", w, h, len(xs), len(ys))
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            tile = img.crop((x, y, min(x + TILE, w), min(y + TILE, h)))
            tw, th = tile.size
            padded = _pad_to_multiple(tile)
            out = _run(pipe, padded, prompt, strength, steps, seed + iy * 97 + ix)
            out = out.crop((0, 0, tw, th))
            wx = _ramp(tw, OVERLAP, x == 0, x + tw >= w)
            wy = _ramp(th, OVERLAP, y == 0, y + th >= h)
            mask = np.outer(wy, wx)[:, :, None]
            acc[y:y + th, x:x + tw] += np.asarray(out, dtype=np.float32) * mask
            weight[y:y + th, x:x + tw] += mask
    weight = np.maximum(weight, 1e-6)
    result = (acc / weight).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")
