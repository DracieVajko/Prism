#!/usr/bin/env python3
"""Unified Standalone Server — images + audio in one process, one contract.


Combines DiffusionServer (:8101) + WhisperApi (:4444) without duplicating
heavy logic: delegates to existing lazy modules (behind the ResidentModelGate
gate), so idle VRAM/RAM stays free. At most one heavy model resident.
Routes (frozen for Odysseus):
GET /health, GET /v1/models, POST /v1/images/generations|edits,
POST /v1/images/inpaint|harmonize|upscale, POST /v1/audio/transcriptions|translations,
GET /v1/images/progress/{id}, GET /webui, GET / -> /webui
"""
import asyncio
import base64
import io
import shutil
import subprocess
import tempfile
import threading
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import (FastAPI, File, Form, Request, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, Response)
from pydantic import BaseModel, Field
from typing import Optional, Tuple, List

from unified.audio_proxy import transcribe_via_service
from unified.audio_splitter import get_audio_duration
from unified.asr import resolve_engine
from unified.ffmpeg_tools import (
    VIDEO_PRESETS, AUDIO_PRESETS, IMAGE_PRESETS, RESIZE_PRESETS,
    convert_video, convert_audio, convert_image, extract_audio, trim_media, compress_video,
    get_media_info, get_presets,
)
from unified.engine import init_diffusion_engine, try_real_inpaint
from unified.resident_model import ResidentModelGate, ModelLoadingError
from unified.action_extractor import ActionExtractor, ActionType
from unified.action_handlers import process_action, process_actions
from unified.ai_client import AIClient, get_ai_client
from unified.registry import build_registry
from unified.transcription_utils import (
    format_transcription,
    split_text_into_chunks,
    merge_transcript_segments,
    validate_audio_file,
    validate_video_file,
    get_media_duration_seconds,
    estimate_transcription_time,
    SUPPORTED_FORMATS,
)
from unified.upscale import (ESRGAN_MODELS, SWINIR_MODELS, ZIMAGE_MODELS,
                             upscale_image_b64)
from unified.vlm import (MODEL_ID as VLM_ID, caption_image_b64, EngineMissing,
                         release as release_vlm)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional bearer-token gate. The product is local-first, so it is off by
# default; setting PRISM_API_TOKEN turns it on for every /v1 route except
# /health. This is the supported way to expose Prism to a LAN or Tailscale.
API_TOKEN = os.environ.get("PRISM_API_TOKEN", "").strip()

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WHISPER_SERVICE_URL = os.getenv("UNIFIED_WHISPER_URL", "http://127.0.0.1:4444")
_manager = ResidentModelGate()
_START_TS = time.time()


IDLE_UNLOAD_SECONDS = int(os.getenv("UNIFIED_IDLE_UNLOAD_SECONDS", "900"))


def _release_engines(reason: str = ""):
    """Release real weights on switch/idle (the manager keeps bookkeeping only)."""
    try:
        import diffusion_server as ds
        if getattr(ds, "_pipe", None) is not None:
            ds.unload_current_model()
    except Exception:
        pass
    try:
        from unified.lcm_onnx_numpy import release as release_pipeline
        release_pipeline()
    except Exception:
        pass
    try:
        from unified.upscale import release as release_upscale
        release_upscale()
    except Exception:
        pass
    try:
        from unified.zimage import release as release_zimage
        release_zimage()
    except Exception:
        pass
    try:
        release_whisper_cache()
    except Exception:
        pass
    try:
        release_vlm()
    except Exception:
        pass


def should_idle_unload(mm, limit: int) -> bool:
    """Idle unload only when a model is resident, nothing runs, and the limit is exceeded."""
    if mm.status()["loaded_kind"] is None:
        return False
    if mm.in_flight() > 0:
        return False
    return mm.idle_seconds() >= limit


async def _idle_unloader():
    while True:
        await asyncio.sleep(30)
        try:
            if not should_idle_unload(_manager, IDLE_UNLOAD_SECONDS):
                continue
            logger.info("idle unload after %.0fs (limit %ss)",
                        _manager.idle_seconds(), IDLE_UNLOAD_SECONDS)
            _manager.unload("idle")
        except Exception as e:
            logger.warning("idle unloader: %s", e)


@asynccontextmanager
async def lifespan(application):
    # Best effort: registry without weights; real inference only with torch + weights.
    try:
        if os.getenv("UNIFIED_NO_ENGINE", "") != "1":
            init_diffusion_engine()
    except Exception as e:
        logger.warning(f"engine init skipped: {e}")
    _manager.on_unload = _release_engines
    if API_TOKEN:
        logger.info("API bearer-token authentication enabled")
    else:
        logger.warning(
            "SECURITY: Prism API has NO authentication. "
            "Bind to 127.0.0.1 (default) for local use only, or set "
            "PRISM_API_TOKEN to require Authorization: Bearer <token>. "
            "LAN access (--host 0.0.0.0 / -Remote) without a token exposes "
            "an unauthenticated API; do NOT expose port %s to the internet.",
            os.getenv("PORT", "8101"),
        )
    task = asyncio.create_task(_idle_unloader())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Unified Standalone Server", lifespan=lifespan)

# Document the error responses the handlers actually return so the OpenAPI
# contract (and Schemathesis) treats them as expected instead of undocumented.
def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    schema = get_openapi(
        title=app.title, version="1.0.0", routes=app.routes,
    )
    _ERROR_RESPONSES = {
        "400": {"description": "Bad request / invalid input"},
        "404": {"description": "Not found"},
        "413": {"description": "Payload too large"},
        "415": {"description": "Unsupported media type"},
        "422": {"description": "Validation error"},
        "500": {"description": "Internal server error"},
        "503": {"description": "Service unavailable (model/engine not loaded)"},
    }
    for _path, _methods in (schema.get("paths") or {}).items():
        for _method, _spec in (_methods or {}).items():
            if not isinstance(_spec, dict):
                continue
            _responses = _spec.setdefault("responses", {})
            for _code, _desc in _ERROR_RESPONSES.items():
                _responses.setdefault(_code, _desc)
    app.openapi_schema = schema
    return schema

app.openapi = _custom_openapi

_TRACKED_PREFIXES = ("/v1/images/", "/v1/audio/", "/v1/chat/")


@app.middleware("http")
async def _require_token(request, call_next):
    """Require the configured bearer token for /v1 routes.

    /health stays open so process supervisors can probe the server; the WebUI
    and static assets stay open because they are local files.
    """
    path = request.url.path
    if (not API_TOKEN or request.method == "OPTIONS"
            or path == "/health" or not path.startswith("/v1/")):
        return await call_next(request)
    if request.headers.get("authorization", "") != f"Bearer {API_TOKEN}":
        return JSONResponse(status_code=401, content={
            "error": "unauthorized",
            "detail": "send Authorization: Bearer $PRISM_API_TOKEN",
        })
    return await call_next(request)


@app.middleware("http")
async def _track_inflight(request, call_next):
    """Counts in-flight inference requests so the idle unloader cannot
    evict a model during a long generation (cold load ~3 min > idle limit)."""
    path = request.url.path
    track = request.method == "POST" and path.startswith(_TRACKED_PREFIXES)
    if track:
        _manager.begin()
    try:
        return await call_next(request)
    finally:
        if track:
            _manager.end()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- helpers ----------

def _placeholder_png_b64(size=(64, 64), color=(200, 30, 30)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        from unified.onnx_env import available_providers
        provs = available_providers()
        if "CUDAExecutionProvider" in provs:
            return "cuda"
        if "DmlExecutionProvider" in provs:
            return "directml"
    except Exception:
        pass
    return "cpu"


def _onnx_provider_label():
    try:
        from unified.onnx_env import available_providers, provider_label, pick_providers
        return provider_label(pick_providers("qdq"))
    except Exception:
        return "unknown"


def _image_registry():
    try:
        return build_registry()
    except Exception:
        return {"image_models": {}, "default_image_model": "DreamShaper_8",
                "audio_models": ["tiny", "base", "small", "medium", "turbo", "large"],
                "default_audio_model": "small"}


def _engines_disabled() -> bool:
    v = os.environ.get("UNIFIED_NO_ENGINE", "")
    return v not in ("", "0", "false", "False", "no")


def _raise_loading(e: ModelLoadingError):
    return JSONResponse(
        status_code=503,
        content={"error": "loading model", "loading_model": e.loading_model,
                 "retry_after": e.retry_after},
    )


# ---------- models ----------

class GenRequest(BaseModel):
    model: Optional[str] = ""
    prompt: str = ""
    negative_prompt: str = ""
    guidance: float = 1.0
    n: int = 1
    size: str = "768x768"
    quality: str = "medium"
    response_format: str = "b64_json"
    request_id: Optional[str] = ""


class InpaintRequest(BaseModel):
    image: str = ""
    mask: Optional[str] = ""
    prompt: str = ""
    negative_prompt: str = ""
    guidance: float = 1.0
    model: Optional[str] = ""
    strength: float = 0.75
    steps: int = 0
    mode: str = "add"          # add | remove | color
    color: str = "#0066ff"     # target color for mode=color
    width: Optional[int] = None
    height: Optional[int] = None
    request_id: Optional[str] = ""


class OutpaintRequest(BaseModel):
    """Expand the canvas and generate the new border with the inpaint engine."""
    image: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    model: Optional[str] = ""
    # Pixels per side; `margin` (fraction of the shorter side) is used for the
    # sides that are left at 0.
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0
    margin: float = 0.0
    guidance: float = 1.0
    strength: float = 0.99
    steps: int = 6
    request_id: Optional[str] = ""


class HarmonizeRequest(BaseModel):
    image: str = ""
    prompt: str = ""
    model: Optional[str] = ""
    color_match: bool = True
    seam_fix: bool = True
    body_mask: Optional[str] = ""
    seam_mask: Optional[str] = ""
    mask: Optional[str] = ""
    strength: float = 0.5
    max_side: int = 1024
    steps: int = 0
    request_id: Optional[str] = ""


class UpscaleRequest(BaseModel):
    image: str = ""
    scale: int = 2
    model: Optional[str] = "RealESRGAN_x4v3"
    # Z-Image Turbo only; ignored by Real-ESRGAN and SwinIR.
    prompt: Optional[str] = ""
    strength: float = 0.33
    seed: int = 0
    request_id: Optional[str] = ""

class ColorizeRequest(BaseModel):
    image: str = ""
    model: Optional[str] = "ddcolor-full"
    request_id: Optional[str] = ""

class RestoreRequest(BaseModel):
    image: str = ""
    prompt: Optional[str] = "restore and colorize this old photo, remove damage, enhance clarity, natural daylight colors, preserve original faces and clothing"
    strength: float = 0.5
    steps: int = 20
    guidance: float = 7.5
    image_guidance: float = 1.5
    request_id: Optional[str] = ""

class FaceRestoreRequest(BaseModel):
    image: str = ""
    request_id: Optional[str] = ""

class SwinIRRequest(BaseModel):
    image: str = ""
    request_id: Optional[str] = ""


class CaptionRequest(BaseModel):
    image: str = ""
    prompt: Optional[str] = "Describe the image in detail."
    max_tokens: int = 256
    model: Optional[str] = ""
    request_id: Optional[str] = ""


def _release_vlm_best_effort():
    try:
        release_vlm()
    except Exception:
        pass


def _release_sd_best_effort():
    try:
        import diffusion_server as ds
        if getattr(ds, "_pipe", None) is not None:
            ds.unload_current_model()
    except Exception:
        pass




from contextlib import contextmanager


@contextmanager
def _in_flight():
    _manager.begin()
    try:
        yield
    finally:
        _manager.end()


# ---------- system ----------

@app.get("/")
def root():
    return RedirectResponse(url="/webui")


@app.get("/health")
def health():
    reg = _image_registry()
    st = _manager.status()
    actual_model, actual_kind = "", None
    try:
        import diffusion_server as ds
        if getattr(ds, "_pipe", None) is not None and getattr(ds, "_model_id", ""):
            actual_model, actual_kind = ds._model_id, "image"
    except Exception:
        pass
    if not actual_model and st["loaded_kind"] == "vlm":
        actual_model, actual_kind = VLM_ID, "vlm"
    if not actual_model and st["loaded_kind"] == "audio":
        actual_model, actual_kind = st["loaded_id"], "audio"
    # faster-whisper weights live in the media module's cache, not in the
    # manager, so report what is actually resident instead of bookkeeping.
    whisper_name = ""
    try:
        from unified.media_routes import _WHISPER_CACHE
        if _WHISPER_CACHE:
            src = str(next(iter(_WHISPER_CACHE))[0])
            whisper_name = Path(src).name.replace("faster-whisper-", "") or src
    except Exception:
        whisper_name = ""
    return {
        "status": "ok",
        "model": reg.get("default_image_model", ""),
        "current_model": actual_model or st["loaded_id"],
        "device": _device(),
        "onnx_provider": _onnx_provider_label(),
        "available_models": sorted(list(reg.get("image_models", {}).keys())),
        "loading": st["loading"],
        "loading_model": st["loading_model"],
        "busy": st["busy"],
        "queue": st["queue"],
        "in_flight": st.get("in_flight"),
        "idle_seconds": st.get("idle_seconds"),
        "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
        "uptime_s": round(time.time() - _START_TS, 1),
        # audio sekcia (WhisperApi kompat)
        "audio_model": whisper_name or reg.get("default_audio_model", "turbo"),
        "audio_loaded": bool(whisper_name) or st["loaded_kind"] == "audio",
        "audio_whisper_models_resident": len(_WHISPER_CACHE),
        "version": "unified-0.1.0",
    }


@app.get("/v1/models")
def list_models():
    reg = _image_registry()
    st = _manager.status()
    models = reg.get("image_models", {})
    data = []
    # The installable catalog is the source of truth for `installed` and kind;
    # build_registry's `missing` flag is not always populated.
    try:
        from unified import model_catalog as _mm
    except Exception:
        _mm = None
    try:
        cat = {m["id"]: m for m in _mm.catalog()} if _mm else {}
    except Exception:
        cat = {}
    for k in sorted(models.keys()):
        entry = models.get(k, {})
        c = cat.get(k, {})
        installed = bool(c.get("installed", not bool(entry.get("missing"))))
        data.append({
            "id": k, "object": "model", "owned_by": "unified",
            "kind": c.get("kind", entry.get("kind", "image")),
            "label": c.get("label", entry.get("label", k)),
            "engine": c.get("engine", entry.get("engine", "")),
            "installed": installed,
            "available": installed,
            "size_mb": c.get("size_mb", 0),
            "install_state": c.get("install_state", "idle"),
            "install_progress": c.get("install_progress", 0),
            "downloadable": bool(c),
            "has_source": bool(c),
        })
    # Audio: faster-whisper names. The installable manager is the single source
    # of truth (exact model.bin size + valid config.json), so an incomplete
    # download never looks installed and the UI can offer a real button.
    # NOTE: base the list on reg["audio_models"] (the supported sizes), not on
    # the catalog, so an entry is never added twice.
    for a in reg.get("audio_models", []):
        entry = {}
        if _mm is not None:
            try:
                entry = _mm.whisper_entry(a)
            except Exception:
                entry = {}
        data.append({
            "id": a, "object": "model", "owned_by": "unified",
            "kind": "audio",
            "label": entry.get("label", f"Whisper {a}"),
            "engine": "faster-whisper",
            "installed": bool(entry.get("installed")),
            "available": True,
            "size_mb": entry.get("size_mb", 0),
            "install_state": entry.get("install_state", "idle"),
            "install_progress": entry.get("install_progress", 0),
            "downloadable": bool(entry),
            "has_source": bool(entry),
        })
    # NVIDIA Nemotron 3.5 ASR: a real alternative to faster-whisper (40
    # languages, native punctuation). Installed state is checked on disk.
    try:
        from unified import nemotron_asr as _nemo
        data.append({
            "id": _nemo.MODEL_ID, "object": "model", "owned_by": "nvidia",
            "kind": "audio",
            "label": "Nemotron 3.5 ASR 0.6B (40 languages, punctuated)",
            "engine": "nemotron-asr",
            "installed": _nemo.is_installed(),
            "available": _nemo.is_installed(),
            "size_mb": 2552,
            "install_state": "idle",
            "install_progress": 0,
            "downloadable": True,
            "has_source": _nemo.is_installed(),
        })
    except Exception:
        pass

    # TTS: Supertonic 3 (multilingual, on-device on PC).
    try:
        from unified import tts_engine as _tts
        data.append({
            "id": "supertonic-3", "object": "model", "owned_by": "supertone",
            "kind": "tts", "label": "Supertonic 3 TTS (31 languages)",
            "engine": "supertonic-3",
            "installed": _tts.is_installed(),
            "available": True,
            "size_mb": 123,
            "install_state": "idle",
            "install_progress": 0,
            "downloadable": True,
            "has_source": _tts.is_installed(),
        })
    except Exception:
        pass

    # Remaining catalog entries (upscale/VLM, plus any audio id not in
    # reg["audio_models"]) that are not already listed.
    known = {d["id"] for d in data}
    for m in cat.values():
        if m.get("id") in known:
            continue
        data.append({
            "id": m["id"], "object": "model", "owned_by": "unified",
            "kind": m.get("kind", "image"),
            "label": m.get("label", m["id"]),
            "engine": m.get("engine", ""),
            "installed": bool(m.get("installed")),
            "available": bool(m.get("installed")),
            "size_mb": m.get("size_mb", 0),
            "install_state": m.get("install_state", "idle"),
            "install_progress": m.get("install_progress", 0),
        })
    return {
        "object": "list",
        "data": data,
        "currently_loaded_model": st["loaded_id"],
        "loading": st["loading"],
        "loading_model": st["loading_model"],
        "missing_models": [k for k, v in models.items() if v.get("missing")],
        "default_image_model": reg.get("default_image_model", ""),
        "default_audio_model": reg.get("default_audio_model", "small"),
        "audio_models": reg.get("audio_models", []),
    }


@app.get("/v1/images/progress/{request_id}")
def image_progress(request_id: str):
    try:
        import diffusion_server as ds
        prog = getattr(ds, "_PROGRESS", {}).get(request_id)
        if prog:
            return {"id": request_id, **prog}
    except Exception:
        pass
    return {"id": request_id, "status": "unknown", "step": 0, "total": 0, "percent": 0}


# ---------- images ----------

def _resolve_image_model(requested):
    reg = _image_registry()
    if requested and requested in reg.get("image_models", {}):
        return requested
    return reg.get("default_image_model", "") or (requested or "")


def _try_real_generation(model_id, prompt, size, quality, negative_prompt="", guidance=1.0, seed=1):
    """Try torch diffusers, then the ONNX LCM pipeline.

    Returns (b64, engine) so the response never hides which engine produced the
    pixels; (None, "") means the caller has to use a placeholder.
    """
    last_err = ""
    try:
        import diffusion_server as ds
        req_id = f"unified-{uuid.uuid4().hex[:8]}"
        req = ds.ImageRequest(model=model_id, prompt=prompt, size=size,
                              quality=quality, n=1, request_id=req_id)
        # ensure_model_loaded does its own lock + 503 on load; just delegate here
        pixels = 0
        try:
            w, h = size.lower().split("x")
            pixels = int(w) * int(h)
        except Exception:
            pass
        ds.ensure_model_loaded(model_id, pixels)
        out = ds._generate_locked(req)
        imgs = out.get("data", [])
        if imgs and imgs[0].get("b64_json"):
            return imgs[0]["b64_json"], "diffusers"
    except Exception as e:
        last_err = str(e)[:160]
        logger.warning(f"real generation fallback: {last_err}")

    # ONNX LCM fallback (no torch/diffusers required)
    try:
        from unified.lcm_onnx_numpy import find_lcm_dir, generate_b64
        d = find_lcm_dir()
        if d is not None:
            try:
                w, h = size.lower().split("x")
                side = max(256, min(512, max(int(w), int(h))))
            except Exception:
                side = 512
            q = (quality or "medium").lower()
            steps = 8 if q == "high" else (6 if q == "medium" else 4)
            return generate_b64(d, prompt, negative_prompt=negative_prompt,
                                guidance=float(guidance or 1.0), size=side,
                                steps=steps, seed=seed), "onnx-lcm"
    except Exception as e:
        last_err = f"{last_err} | onnx: {str(e)[:120]}"
        logger.warning(f"onnx generation fallback failed: {str(e)[:200]}")
    return None, (f"failed: {last_err}" if last_err else "")


@app.post("/v1/images/generations")
def generations(req: GenRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    model_id = _resolve_image_model(req.model)
    try:
        _manager.ensure("image", model_id)
    except ModelLoadingError as e:
        return _raise_loading(e)
    _release_vlm_best_effort()
    n = max(1, min(4, req.n or 1))
    data = []
    engines = []
    for i in range(n):
        b64, engine = _try_real_generation(model_id, req.prompt, req.size or "768x768",
                                           req.quality or "medium",
                                           req.negative_prompt, req.guidance,
                                           seed=int(time.time()) + i * 7919)
        if not b64:
            b64 = _placeholder_png_b64()
            engine = engine or "placeholder"
        data.append({"b64_json": b64})
        engines.append(engine)
    try:
        w, h = (req.size or "768x768").lower().split("x")
        side = max(256, min(512, max(int(w), int(h))))
    except Exception:
        side = 512
    q = (req.quality or "medium").lower()
    steps = 8 if q == "high" else (6 if q == "medium" else 4)
    # Report the size of the image that was actually produced (the torch path
    # honours the request, the ONNX fallback generates a square at `side`).
    real_size = f"{side}x{side}"
    try:
        from PIL import Image as _Image
        _im = _Image.open(io.BytesIO(base64.b64decode(data[0]["b64_json"])))
        real_size = f"{_im.width}x{_im.height}"
    except Exception:
        pass
    stored = []
    for item in data:
        meta = _store_b64("image", ".png", item.get("b64_json", ""), rid)
        if meta:
            stored.append(meta)
            item["output"] = meta
    return {
        "created": int(time.time()),
        "data": data,
        "outputs": stored,
        "request_id": rid,
        "size": real_size,
        "requested_size": req.size or "768x768",
        "quality": q,
        "steps": steps,
        "engine": engines[0] if len(set(engines)) == 1 else "mixed",
        "engines": engines,
    }


@app.post("/v1/images/edits")
async def edits(
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
    model: str = Form(default=""),
    n: int = Form(default=1),
    size: str = Form(default="768x768"),
    quality: str = Form(default="medium"),
    strength: float = Form(default=0.75),
    request_id: str = Form(default=""),
):
    rid = request_id or f"req-{uuid.uuid4().hex[:8]}"
    model_id = _resolve_image_model(model)
    try:
        _manager.ensure("image", model_id)
    except ModelLoadingError as e:
        return _raise_loading(e)
    _release_vlm_best_effort()
    # img2img: prefer the selected diffusion model (honours the requested size
    # and actually uses the model the user picked), then the ONNX LCM engine.
    raw = await image.read()
    b64 = None
    engine = None
    if raw:
        try:
            import diffusion_server as ds
            try:
                _w, _h = size.lower().split("x")
                _pixels = int(_w) * int(_h)
            except Exception:
                _pixels = 0
            with ds._tracked_model_lock():
                ds.ensure_model_loaded(model_id, _pixels)
                res = await ds._run_edit(prompt or "edit", raw, 1, size, quality,
                                         strength, "b64_json", rid)
            payload = (res.get("data") or [{}])[0].get("b64_json")
            if payload:
                b64 = payload
                engine = "diffusers"
            elif res.get("error"):
                logger.warning(f"diffusers edit unavailable: {str(res.get('error'))[:120]}")
        except Exception as e:
            logger.warning(f"diffusers edit failed: {str(e)[:160]}")
    try:
        from unified.lcm_onnx_numpy import find_lcm_dir, img2img_b64
        d = find_lcm_dir()
        if b64 is None and d is not None and raw:
            import base64 as _b64
            try:
                w, h = size.lower().split("x")
                side = max(256, min(512, max(int(w), int(h))))
            except Exception:
                side = 512
            steps = 6 if (quality or "medium") == "medium" else 4
            b64 = img2img_b64(d, _b64.b64encode(raw).decode(), prompt or "edit",
                              strength=float(strength or 0.75), size=side, steps=steps)
            engine = "onnx-lcm"
    except Exception as e:
        logger.warning(f"onnx img2img failed: {str(e)[:200]}")
    if not b64:
        b64, engine = _try_real_generation(model_id, prompt or "edit", size, quality)
    if not b64:
        b64 = _placeholder_png_b64(color=(30, 120, 200))
        engine = "placeholder"
    real_size = ""
    try:
        from PIL import Image as _Image
        _im = _Image.open(io.BytesIO(base64.b64decode(b64)))
        real_size = f"{_im.width}x{_im.height}"
    except Exception:
        pass
    meta = _store_b64("image", ".png", b64, rid)
    return {"created": int(time.time()), "data": [{"b64_json": b64}],
            "request_id": rid, "engine": engine, "size": real_size,
            "outputs": [meta] if meta else []}


@app.post("/v1/images/inpaint")
def inpaint(req: InpaintRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    model_id = _resolve_image_model(req.model)
    try:
        _manager.ensure("image", model_id)
    except ModelLoadingError as e:
        return _raise_loading(e)
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    t0 = time.time()
    _release_vlm_best_effort()
    # One big engine at a time on a 6 GB card: a resident diffusers pipeline
    # starved the ONNX inpaint and the request hung for minutes.
    _release_sd_best_effort()
    # Real pipeline when the engine is loaded (masks = removing people in the background),
    # otherwise echo the input — the contract {image, elapsed, request_id} always holds.
    if req.mask:
        real = None
        try:
            from unified.lcm_onnx_numpy import find_lcm_dir, inpaint_b64
            d = find_lcm_dir()
            if d is not None:
                try:
                    w, h = (req.width, req.height) if (req.width and req.height) else (512, 512)
                    side = max(256, min(512, max(int(w), int(h))))
                except Exception:
                    side = 512
                steps = req.steps or 6
                real = inpaint_b64(d, req.image, req.mask, req.prompt,
                                   negative_prompt=req.negative_prompt,
                                   guidance=float(req.guidance or 1.0),
                                   strength=float(req.strength or 0.8),
                                   size=side, steps=steps,
                                   mode=req.mode or "add", color=req.color or "#0066ff")
                meta = _store_b64("image", ".png", real, rid)
                return {"image": real, "elapsed": round(time.time() - t0, 2),
                        "request_id": rid, "engine": "onnx-inpaint",
                        "outputs": [meta] if meta else []}
        except Exception as e:
            logger.warning(f"onnx inpaint failed: {str(e)[:200]}")
        try:
            real = try_real_inpaint(req.image, req.mask, req.prompt, model_id,
                                    req.strength, req.steps)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)[:200]})
        except Exception as e:
            logger.warning(f"real inpaint failed, echo fallback: {str(e)[:160]}")
            real = None
        if real:
            meta = _store_b64("image", ".png", real, rid)
            return {"image": real, "elapsed": round(time.time() - t0, 2),
                    "request_id": rid, "engine": "diffusion",
                    "outputs": [meta] if meta else []}
    return {"image": req.image, "elapsed": round(time.time() - t0, 2),
            "request_id": rid, "engine": "echo"}


MAX_OUTPAINT_SIDE = 2048
MAX_OUTPAINT_PIXELS = 4_000_000


def _edge_pad_image(img, left: int, right: int, top: int, bottom: int):
    """Pad with replicated edge pixels so the model has real context to extend."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(img.convert("RGB"))
    padded = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="edge")
    return Image.fromarray(padded, "RGB")


def _border_mask(width: int, height: int, left: int, right: int,
                 top: int, bottom: int):
    import numpy as np
    from PIL import Image

    m = np.zeros((height, width), dtype=np.uint8)
    if left:
        m[:, :left] = 255
    if right:
        m[:, width - right:] = 255
    if top:
        m[:top, :] = 255
    if bottom:
        m[height - bottom:, :] = 255
    return Image.fromarray(m, "L")


@app.post("/v1/images/outpaint")
def outpaint(req: OutpaintRequest):
    """Expand the canvas and let the inpaint model paint the new border."""
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    model_id = _resolve_image_model(req.model)
    try:
        _manager.ensure("image", model_id)
    except ModelLoadingError as e:
        return _raise_loading(e)
    try:
        import numpy as np  # noqa: F401
        from unified.lcm_onnx_numpy import (_b64_image_to_pil, _pil_to_b64,
                                            find_lcm_dir, inpaint_b64)
    except Exception as e:
        return JSONResponse(status_code=503, content={
            "error": "will not load", "detail": f"outpaint engine unavailable: {str(e)[:160]}"})
    d = find_lcm_dir()
    if d is None:
        return JSONResponse(status_code=503, content={
            "error": "will not load",
            "detail": "LCM ONNX pipeline not found; install LCM_Dreamshaper_ONNX"})

    try:
        src = _b64_image_to_pil(req.image).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad image b64: {str(e)[:120]}"})

    w, h = src.size
    short = min(w, h)
    left = int(req.left or 0)
    right = int(req.right or 0)
    top = int(req.top or 0)
    bottom = int(req.bottom or 0)
    if req.margin and req.margin > 0:
        m = int(round(short * float(req.margin)))
        left = left or m
        right = right or m
        top = top or m
        bottom = bottom or m
    if min(left, right, top, bottom) < 0:
        return JSONResponse(status_code=400, content={"error": "negative margin"})
    if not any((left, right, top, bottom)):
        return JSONResponse(status_code=400, content={
            "error": "nothing to expand: pass left/right/top/bottom pixels or margin"})
    new_w, new_h = w + left + right, h + top + bottom
    if new_w > MAX_OUTPAINT_SIDE or new_h > MAX_OUTPAINT_SIDE:
        return JSONResponse(status_code=400, content={
            "error": f"result too large: {new_w}x{new_h} (max side {MAX_OUTPAINT_SIDE})"})
    if new_w * new_h > MAX_OUTPAINT_PIXELS:
        return JSONResponse(status_code=400, content={
            "error": f"result too large: {new_w}x{new_h} (max {MAX_OUTPAINT_PIXELS} pixels)"})

    t0 = time.time()
    _release_sd_best_effort()
    padded = _edge_pad_image(src, left, right, top, bottom)
    mask = _border_mask(new_w, new_h, left, right, top, bottom)
    size = max(256, min(512, max(new_w, new_h)))
    try:
        out_b64 = inpaint_b64(d, _pil_to_b64(padded), _pil_to_b64(mask),
                              req.prompt, negative_prompt=req.negative_prompt,
                              guidance=float(req.guidance or 1.0),
                              strength=float(req.strength or 0.99),
                              size=int(size), steps=int(req.steps or 6),
                              mode="add")
    except Exception as e:
        logger.warning(f"outpaint failed: {str(e)[:200]}")
        return JSONResponse(status_code=500, content={"error": f"outpaint failed: {str(e)[:200]}"})
    meta = _store_b64("image", ".png", out_b64, rid)
    return {"image": out_b64, "elapsed": round(time.time() - t0, 2),
            "outputs": [meta] if meta else [],
            "request_id": rid, "engine": "onnx-outpaint",
            "input_size": [w, h], "output_size": [new_w, new_h],
            "margins": {"left": left, "right": right, "top": top, "bottom": bottom}}


@app.post("/v1/images/harmonize")
def harmonize(req: HarmonizeRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    model_id = _resolve_image_model(req.model)
    try:
        _manager.ensure("image", model_id)
    except ModelLoadingError as e:
        return _raise_loading(e)
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    t0 = time.time()
    mask = req.mask or req.body_mask or req.seam_mask or ""
    try:
        from unified.lcm_onnx_numpy import harmonize_b64
        out = harmonize_b64(req.image, mask, color_match=req.color_match,
                            seam_fix=req.seam_fix, strength=req.strength)
        return {"image": out, "elapsed": round(time.time() - t0, 2),
                "request_id": rid, "engine": "cpu-harmonize"}
    except Exception as e:
        logger.warning(f"harmonize failed: {str(e)[:200]}")
        return {"image": req.image, "elapsed": round(time.time() - t0, 2),
                "request_id": rid, "engine": "echo"}


@app.post("/v1/images/upscale")
def upscale(req: UpscaleRequest):
    logger.error("DEBUG upscale globals _store_b64: %s", "_store_b64" in globals())
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    model_id = (req.model or "RealESRGAN_x4v3").strip()
    try:
        _manager.ensure("upscale", model_id)
    except ModelLoadingError as e:
        return _raise_loading(e)
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    _release_sd_best_effort()
    try:
        out = upscale_image_b64(req.image, req.scale, model_id,
                                prompt=req.prompt or "",
                                strength=req.strength, seed=req.seed)
    except RuntimeError as e:
        # Z-Image not installed or no CUDA device: honest 503, never fake output.
        hint = ("Run: python download_models.py z-image-turbo"
                if model_id.lower() in ("zimage", "zimage-turbo", "z-image-turbo")
                else "Run: python download_models.py swinir-realsr-x4-upscale")
        return JSONResponse(status_code=503, content={
            "error": "will not load", "detail": str(e)[:300], "hint": hint})
    except ValueError as e:
        msg = str(e)
        if "exceeds cap" in msg:
            return JSONResponse(status_code=503, content={"error": "will not load", "detail": msg})
        return JSONResponse(status_code=400, content={"error": msg})
    out["request_id"] = rid
    meta = _store_b64("image", ".png", out.get("image", ""), rid)
    out["outputs"] = [meta] if meta else []
    return out


VIDEO_MAX_FRAMES_DEFAULT = 900
VIDEO_MAX_FRAMES = {"z-image-turbo": 24, "zimage": 24, "z-image": 24,
                    "zimage-turbo": 24}


def _rate_to_float(value: str) -> float:
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den or 1)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0



# --- media routes moved to unified.media_routes ---
@app.get("/v1/models/catalog")
def models_catalog():
    """Catalog of installable models + state (installed/downloading/…)."""
    try:
        return {
            "models": _mm.catalog(),
            "kinds": [
                {"id": k, "label": v} for k, v in _mm.KIND_LABELS.items()
            ],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})


@app.post("/v1/models/install")
def models_install(req: InstallRequest):
    try:
        return _mm.start_install(req.id)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e).strip("'")})


@app.post("/v1/models/uninstall")
def models_uninstall(req: InstallRequest):
    try:
        return _mm.uninstall(req.id)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e).strip("'")})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})


@app.get("/v1/models/progress/{model_id}")
def models_progress(model_id: str):
    """Same shape as Android /v1/models/progress/{id}."""
    try:
        return _mm.progress(model_id)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e).strip("'")})


@app.get("/v1/models/status/{model_id}")
def models_status(model_id: str):
    try:
        return _mm.status(model_id)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e).strip("'")})


@app.post("/v1/models/verify/{model_id}")
def models_verify(model_id: str):
    try:
        return _mm.verify(model_id)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e).strip("'")})


@app.post("/v1/models/cancel")
def models_cancel(req: InstallRequest):
    try:
        return _mm.cancel(req.id)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e).strip("'")})




@app.get("/v1/settings")
def settings_info(request: Request):
    """Runtime info for the WebUI Settings tab (read-only, no secrets)."""
    import shutil as _sh
    torch_ok = False
    cuda = False
    try:
        import torch
        torch_ok = True
        cuda = torch.cuda.is_available()
    except Exception:
        pass
    onnx_ok = False
    try:
        import onnxruntime  # noqa: F401
        onnx_ok = True
    except Exception:
        pass
    whisper_ok = False
    try:
        import faster_whisper  # noqa: F401
        whisper_ok = True
    except Exception:
        pass
    return {
        "version": "unified-0.1.0",
        # The port the request actually arrived on, not a hardcoded default.
        "port": request.url.port or 8101,
        "whisper_url": WHISPER_SERVICE_URL,
        "whisper_mode": "local (faster-whisper)" if whisper_ok else "remote",
        "models_dir": str(HERE / "models"),
        "output_dir": str(HERE / "output"),
        "default_image_model": _image_registry().get("default_image_model", ""),
        "audio_models": list(ALLOWED_WHISPER),
        "engines": {
            "torch": torch_ok,
            "cuda": cuda,
            "onnxruntime": onnx_ok,
            "whisper_native": whisper_ok,
            "ffmpeg": _sh.which("ffmpeg") or "",
            "ffprobe": _sh.which("ffprobe") or "",
            "ytdlp": _sh.which("yt-dlp") or "",
        },
    }




# ---------- model finder / custom models ----------

from unified import model_finder as _mf


class AddModelRequest(BaseModel):
    repo_id: str = ""
    name: str = ""
    url: str = ""
    kind: str = "custom"


class DeleteModelRequest(BaseModel):
    name: str = ""


@app.get("/v1/models/find")
def models_find(q: str = "", limit: int = 20, task: str = ""):
    try:
        return _mf.search_hf(q, limit=limit, filter_task=task)
    except RuntimeError as e:
        return JSONResponse(status_code=502, content={"error": str(e)[:300]})


@app.get("/v1/models/custom")
def models_custom():
    res = _mf.custom_models()
    for m in res["models"]:
        m.update(_mf.add_state(m["name"]))
    return res


@app.post("/v1/models/add")
def models_add(req: AddModelRequest):
    try:
        if req.url.strip():
            return _mf.start_add_url(req.url, req.name, req.kind)
        return _mf.start_add(req.repo_id, req.name)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/v1/models/add/status/{name}")
def models_add_status(name: str):
    return {"name": name, **_mf.add_state(name)}


@app.post("/v1/models/delete")
def models_delete(req: DeleteModelRequest):
    try:
        return _mf.delete_custom(req.name)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e).strip("'")})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})


from unified.media_routes import (router as media_router, _output_path,
                                 _store_b64, _transcribe_with_vad,
                                 _transcribe_with_whisper, transcriptions,
                                 _chat_vision_parts, _transcribe_pcm,
                                 transcribe_alias)

logger.error("DEBUG startup _store_b64 in globals: %s", "_store_b64" in globals())
logger.error("DEBUG startup media_routes file: %s", __import__("unified.media_routes", fromlist=["__file__"]).__file__)
app.routes.extend(media_router.routes)
_output_path = _output_path  # keep test/compatibility alias
_store_b64 = _store_b64
_transcribe_with_vad = _transcribe_with_vad
_transcribe_with_whisper = _transcribe_with_whisper
transcriptions = transcriptions
_chat_vision_parts = _chat_vision_parts
_transcribe_pcm = _transcribe_pcm
transcribe_alias = transcribe_alias


@app.on_event("startup")
def _debug_store_b64():
    logger.error("DEBUG startup _store_b64 in globals: %s", "_store_b64" in globals())
    logger.error("DEBUG startup upscale globals: %s", "_store_b64" in upscale.__globals__)
    logger.error("DEBUG startup upscale func id: %s", id(upscale))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("unified_server:app", host="127.0.0.1", port=8101, workers=1)
