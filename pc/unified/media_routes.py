"""Media routes: video upscale, ffmpeg helpers, outputs, probe.

Extracted from unified_server.py to keep the composition root readable.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import numpy as np

logger = logging.getLogger(__name__)
from pathlib import Path
from unified.audio_splitter import get_audio_duration
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field


class UpscaleRequest(BaseModel):
    image: str = ""
    scale: int = 2
    model: str | None = "RealESRGAN_x4v3"
    prompt: str | None = ""
    strength: float = 0.33
    seed: int = 0
    request_id: str | None = ""


class ColorizeRequest(BaseModel):
    image: str = ""
    model: str | None = "deoldify-artistic"
    request_id: str | None = ""


class RestoreRequest(BaseModel):
    image: str = ""
    prompt: str | None = "restore and colorize this old photo, remove damage, enhance clarity, natural daylight colors, preserve original faces and clothing"
    strength: float = 0.5
    steps: int = 20
    guidance: float = 7.5
    image_guidance: float = 1.5
    request_id: str | None = ""


class FaceRestoreRequest(BaseModel):
    image: str = ""
    request_id: str | None = ""


class SwinIRRequest(BaseModel):
    image: str = ""
    request_id: str | None = ""


class CaptionRequest(BaseModel):
    image: str = ""
    prompt: str | None = "Describe the image in detail."
    max_tokens: int = 256
    model: str | None = ""
    request_id: str | None = ""

from unified.ffmpeg_tools import (
    compress_video,
    extract_audio,
    get_media_info,
    get_presets,
    trim_media,
)
from unified.action_extractor import ActionType
from unified.asr import resolve_engine
from unified.audio_proxy import transcribe_via_service
from unified.transcription_utils import format_transcription
from unified.upscale import upscale_image_b64
from unified.vlm import MODEL_ID as VLM_ID
from unified.vlm import EngineMissing, caption_image_b64

router = APIRouter()
HERE = Path(__file__).resolve().parent.parent

# _us is defined below; router must exist before route decorators.
ALLOWED_WHISPER = ("tiny", "base", "small", "medium", "turbo", "large",
                   "ivrit-whisper-turbo")

def _us():
    import unified_server as us
    return us

def _output_path(rel: str) -> Path:
    """Resolve a client-supplied output path, refusing anything outside output/."""
    if not rel:
        raise ValueError("empty output path")
    if len(rel.encode("utf-8")) > MAX_OUTPUT_REL_BYTES:
        raise ValueError("output path too long")
    segments = rel.replace("\\", "/").split("/")
    if any(len(seg.encode("utf-8")) > MAX_OUTPUT_SEGMENT_BYTES
           for seg in segments):
        raise ValueError("output path component too long")
    if any(seg in ("", ".", "..") for seg in segments):
        raise ValueError("invalid output path")
    root = _us().OUTPUT_DIR.resolve()
    try:
        target = (root / rel).resolve()
    except OSError as e:
        raise ValueError(f"invalid output path: {e}") from e
    if root not in target.parents and target != root:
        raise ValueError("path outside output/")
    return target



def _rate_to_float(value: str) -> float:
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den or 1)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


VIDEO_MAX_FRAMES_DEFAULT = 900
VIDEO_MAX_FRAMES = {"z-image-turbo": 24, "zimage": 24, "z-image": 24,
                    "zimage-turbo": 24}

def _probe_video(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,width,height",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "ffprobe failed")[:300])
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or [{}]
    fmt = data.get("format") or {}
    return {
        "fps": _rate_to_float(streams[0].get("avg_frame_rate", "")),
        "width": int(streams[0].get("width") or 0),
        "height": int(streams[0].get("height") or 0),
        "duration": float(fmt.get("duration") or 0),
    }


@router.post("/v1/videos/upscale")
async def video_upscale(
    file: UploadFile = File(default=None),
    scale: int = Form(default=2),
    model: str = Form(default="RealESRGAN_x4v3"),
    fps: float = Form(default=0.0),
    prompt: str = Form(default=""),
    strength: float = Form(default=0.33),
):
    """Upscale a video by extracting frames, upscaling each, and remuxing audio.

    Model choice matters: Real-ESRGAN/SwinIR take well under a second per
    frame; Z-Image is a diffusion refine at tens of seconds per frame, so it is
    capped at 24 frames (about 1 s at 24 fps). Longer clips should use the
    deterministic engines or be split first.
    """
    rid = f"vid-{uuid.uuid4().hex[:8]}"
    if file is None:
        return JSONResponse(status_code=400, content={"error": "missing file"})
    if scale not in (2, 3, 4):
        return JSONResponse(status_code=400,
                            content={"error": f"scale must be 2/3/4, got {scale}"})
    from unified.upscale import ESRGAN_MODELS, SWINIR_MODELS, ZIMAGE_MODELS
    allowed_models = ESRGAN_MODELS | SWINIR_MODELS | ZIMAGE_MODELS
    if (model or "").strip().lower() not in allowed_models:
        return JSONResponse(status_code=400, content={
            "error": f"unknown upscale model '{model}'",
            "detail": "expected RealESRGAN_x4v3, swinir-realsr-x4 or z-image-turbo"})
    raw = await file.read()
    if not raw:
        return JSONResponse(status_code=400, content={"error": "empty file"})
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return JSONResponse(status_code=503, content={
            "error": "will not load", "detail": "ffmpeg/ffprobe not on PATH"})

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmpdir = Path(tempfile.mkdtemp(prefix="prism_video_"))
    t0 = time.time()
    try:
        src = tmpdir / f"input{suffix}"
        src.write_bytes(raw)
        try:
            info = _probe_video(src)
        except Exception as e:
            return JSONResponse(status_code=400, content={
                "error": "invalid or unsupported video",
                "detail": str(e)[:200]})
        proc_fps = float(fps) if fps and fps > 0 else min(info["fps"] or 24.0, 30.0)
        if proc_fps <= 0:
            return JSONResponse(status_code=400,
                                content={"error": "could not detect video fps"})
        frames_dir = tmpdir / "frames"
        frames_dir.mkdir()
        extract = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src),
             "-vf", f"fps={proc_fps:.6g}", "-start_number", "0",
             str(frames_dir / "%06d.png")],
            capture_output=True, text=True)
        if extract.returncode != 0:
            return JSONResponse(status_code=400, content={
                "error": "invalid or unsupported video",
                "detail": (extract.stderr or "frame extraction failed")[:300]})
        frames = sorted(frames_dir.glob("*.png"))
        if not frames:
            return JSONResponse(status_code=400, content={"error": "no video frames"})

        cap = VIDEO_MAX_FRAMES.get(model.strip().lower(), VIDEO_MAX_FRAMES_DEFAULT)
        if len(frames) > cap:
            return JSONResponse(status_code=400, content={
                "error": "too many frames for this model",
                "detail": (f"{len(frames)} frames at {proc_fps:.3g} fps; "
                           f"{model} is capped at {cap}. Lower fps, trim the "
                           "clip, or use RealESRGAN/SwinIR."),
                "frames": len(frames), "max_frames": cap})

        _us()._manager.ensure("upscale", model)
        _us()._release_sd_best_effort()
        engine = ""
        for idx, fp in enumerate(frames, 1):
            b64 = base64.b64encode(fp.read_bytes()).decode()
            out = await asyncio.to_thread(
                upscale_image_b64, b64, scale, model, prompt, strength, 0)
            fp.write_bytes(base64.b64decode(out["image"]))
            engine = out.get("engine", engine)
            if idx % 10 == 0 or idx == len(frames):
                logger.info("video %s: %d/%d frames", rid, idx, len(frames))

        out_path = tmpdir / "out.mp4"
        mux = subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-framerate", f"{proc_fps:.6g}",
             "-i", str(frames_dir / "%06d.png"),
             "-i", str(src),
             "-map", "0:v:0", "-map", "1:a:0?",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
             str(out_path)],
            capture_output=True, text=True)
        if mux.returncode != 0:
            raise RuntimeError((mux.stderr or "remux failed")[:300])
        meta = _store_output("video", ".mp4", out_path.read_bytes(), rid)
        return {
            "request_id": rid,
            "frames": len(frames),
            "fps": round(proc_fps, 3),
            "scale": scale,
            "model": model,
            "engine": engine,
            "elapsed": round(time.time() - t0, 1),
            "outputs": [meta] if meta else [],
            "url": meta.get("url", ""),
        }
    except Exception as e:
        logger.warning("video upscale failed: %s", str(e)[:200])
        return JSONResponse(status_code=500,
                            content={"error": str(e)[:300], "request_id": rid})
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@router.post("/v1/images/colorize")
def colorize(req: ColorizeRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    try:
        from unified.colorize import colorize_b64
        out = colorize_b64(req.image, model_id=req.model or "deoldify-artistic")
    except FileNotFoundError:
        return JSONResponse(status_code=503, content={
            "error": "will not load", "detail": "DDColor model not installed",
            "hint": "Run: python download_models.py deoldify-artistic"})
    except Exception as e:
        logger.warning(f"colorize failed: {str(e)[:200]}")
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})
    out["request_id"] = rid
    meta = _store_b64("image", ".png", out.get("image", ""), rid)
    out["outputs"] = [meta] if meta else []
    return out


@router.post("/v1/images/restore")
async def restore(req: RestoreRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    # Restore uses img2img with a restoration prompt through any available SD model.
    _us()._release_sd_best_effort()
    _us()._release_vlm_best_effort()
    try:
        from unified.ocr_engine import reset_ocr_engine
        reset_ocr_engine()
    except Exception:
        pass
    prompt = req.prompt or (
        "restore this old damaged photo, remove scratches, stains and fading, "
        "keep the original faces, clothing and composition, natural colors, "
        "sharp real detail")
    engine_pref = (os.environ.get("PRISM_RESTORE_ENGINE") or "").strip().lower()

    # Z-Image Turbo keeps the source composition better than SDXL img2img on
    # old prints, so it is the preferred engine when installed. `strength` is
    # kept low: this is a repair, not a redraw.
    zimage_first = engine_pref != "sd"
    if zimage_first:
        try:
            from unified.zimage import is_installed as zimage_installed
            from unified.zimage import refine_b64 as zimage_refine
            if zimage_installed():
                import time as _t
                t0 = _t.time()
                # Turbo is distilled for 8-9 steps; more only adds time.
                z_steps = min(max(int(req.steps or 9), 4), 12)
                out = zimage_refine(req.image, prompt=prompt,
                                    strength=min(max(float(req.strength or 0.45), 0.05), 0.6),
                                    steps=z_steps)
                b64 = out.get("image", "")
                if b64:
                    meta = _store_b64("image", ".png", b64, rid)
                    return {"image": b64, "elapsed": round(_t.time() - t0, 1),
                            "engine": "zimage-turbo", "request_id": rid,
                            "outputs": [meta] if meta else []}
                logger.warning("zimage restore returned no image; falling back to SD")
        except Exception as e:
            logger.warning("zimage restore failed (%s); falling back to SD", str(e)[:200])

    try:
        import base64 as _b64
        import diffusion_server as ds
        ds.ensure_registry()
        ds._args.disable_safety_checker = True
        _us()._release_sd_best_effort()
        _us()._release_vlm_best_effort()
        # SD fallback: SDXL photorealistic (Juggernaut) preserves faces better
        # than SD 1.5. Override with PRISM_RESTORE_MODEL.
        restore_model = (os.environ.get("PRISM_RESTORE_MODEL") or "JuggernautXL_v9").strip()
        reg = ds.ensure_registry()
        if restore_model not in reg or reg[restore_model].get("missing"):
            restore_model = "DreamShaper_8"
        if restore_model not in reg or reg[restore_model].get("missing"):
            # Neither engine is installed: honest 503, not a 500.
            return JSONResponse(status_code=503, content={
                "error": "will not load",
                "detail": "no restoration engine installed "
                          "(Z-Image Turbo FP8 or an SD/SDXL checkpoint)",
                "hint": "Run: python download_models.py z-image-turbo-fp8 "
                        "or --preset sd15",
            })
        ds._apply_profile(restore_model)
        plan = ds._plan_device(restore_model, 768*768)
        ds.load_model(device=plan["device"], dtype=plan["dtype"])
        raw = _b64.b64decode(req.image)
        edit_size = "768x768" if restore_model != "DreamShaper_8" else "512x512"
        # A low strength keeps the subject/composition; 0.5 redrew the scene.
        sd_strength = min(float(req.strength or 0.28), 0.35)
        out = await ds._run_edit(prompt, raw, 1, edit_size, "medium",
                                sd_strength, "b64_json", rid)
        b64 = (out.get("data") or [{}])[0].get("b64_json", "")
        if not b64:
            raise RuntimeError(out.get("error", "restore produced no image"))
        return {"image": b64, "elapsed": 0, "engine": out.get("engine", "diffusers"),
                "request_id": rid, "outputs": out.get("outputs", [])}
    except Exception as e:
        logger.warning(f"restore failed: {str(e)[:200]}")
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})


@router.post("/v1/images/face-restore")
def face_restore(req: FaceRestoreRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    # GFPGAN v1.4 was removed: the ONNX export produced an invalid (pink/orb)
    # image on every test input. Use /v1/images/restore (Juggernaut img2img)
    # or /v1/images/colorize instead of a face-only model that does not work.
    return JSONResponse(status_code=503, content={
        "error": "will not load",
        "detail": "GFPGAN v1.4 was removed after producing invalid output.",
        "hint": "Use POST /v1/images/restore (restoration) or /v1/images/colorize."})
    out["request_id"] = rid
    meta = _store_b64("image", ".png", out.get("image", ""), rid)
    out["outputs"] = [meta] if meta else []
    return out


class SecretsRequest(BaseModel):
    hf_token: str | None = None
    llm: dict | None = None


@router.get("/v1/settings/secrets")
def get_secrets():
    from unified import secrets
    return secrets.public_view()


@router.put("/v1/settings/secrets")
def put_secrets(req: SecretsRequest):
    from unified import secrets
    payload = {}
    if req.hf_token is not None:
        payload["hf_token"] = req.hf_token
    if req.llm is not None:
        payload["llm"] = req.llm
    return secrets.save(payload)


@router.post("/v1/settings/secrets/test/hf")
def test_hf():
    from unified import secrets
    return secrets.test_hf_token()


@router.post("/v1/settings/secrets/test/llm")
def test_llm():
    from unified import secrets
    return secrets.test_llm()


@router.get("/v1/update/check")
def check_update():
    from unified.update_checker import check_update as _check
    return _check()

@router.post("/v1/update/apply")
def apply_update():
    from unified.update_checker import apply_update as _apply
    return _apply()

@router.post("/v1/images/restore-upscale")
def restore_upscale(req: SwinIRRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    try:
        from unified.swinir import upscale_b64
        out = upscale_b64(req.image)
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={
            "error": "will not load", "detail": str(e)[:200],
            "hint": "Run: python download_models.py swinir-realsr-x4"})
    except Exception as e:
        logger.warning(f"restore-upscale failed: {str(e)[:200]}")
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})
    out["request_id"] = rid
    meta = _store_b64("image", ".png", out.get("image", ""), rid)
    out["outputs"] = [meta] if meta else []
    return out


# ---------- vision (V2) ----------

@router.post("/v1/images/caption")
def caption(req: CaptionRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    if _us()._engines_disabled():
        return JSONResponse(status_code=503, content={
            "error": "will not load", "detail": "engines disabled by UNIFIED_NO_ENGINE"})
    try:
        _us()._manager.ensure("vlm", VLM_ID)
    except ModelLoadingError as e:
        return _us()._raise_loading(e)
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    _us()._release_sd_best_effort()
    try:
        out = caption_image_b64(req.image, req.prompt or "Describe the image in detail.",
                                req.max_tokens or 256)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    except EngineMissing as e:
        return JSONResponse(status_code=503,
                            content={"error": "will not load", "detail": str(e)[:200]})
    out["request_id"] = rid
    try:
        meta = _store_output("text", ".txt", str(out.get("caption", "")).encode("utf-8"), rid)
        out["outputs"] = [meta]
    except Exception:
        out["outputs"] = []
    return out


def _chat_vision_parts(messages):
    """Extract (text, image_b64) from the last user message in OpenAI messages."""
    text, image = "", ""
    items = messages if isinstance(messages, list) else []
    for m in reversed(items):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content", "")
        if isinstance(c, str):
            if c.strip():
                text = c
                break
        elif isinstance(c, list):
            for part in c:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and not text:
                    text = part.get("text", "")
                elif part.get("type") == "image_url" and not image:
                    u = (part.get("image_url") or {}).get("url", "")
                    if u.startswith("data:image/") and ";base64," in u:
                        image = u.split(";base64,", 1)[1]
            if text or image:
                break
    return text, image


@router.post("/v1/chat/completions")
def chat_completions(body: dict):
    model = (body.get("model") or VLM_ID) if isinstance(body, dict) else VLM_ID
    messages = body.get("messages", []) if isinstance(body, dict) else []
    max_tokens = body.get("max_tokens", 256) if isinstance(body, dict) else 256
    if _us()._engines_disabled():
        return JSONResponse(status_code=503, content={
            "error": "will not load", "detail": "engines disabled by UNIFIED_NO_ENGINE"})
    try:
        _us()._manager.ensure("vlm", VLM_ID)
    except ModelLoadingError as e:
        return _us()._raise_loading(e)
    text, image = _chat_vision_parts(messages)
    # Text-only path: route to an external LLM provider (Ollama, LM Studio,
    # llama.cpp, OpenAI, Mistral, Gemini, ...). Vision path uses SmolVLM.
    if not image:
        try:
            from unified.llm_providers import _get_config, process_with_external_llm
            cfg = _get_config()
            if not cfg.get("base_url"):
                return JSONResponse(status_code=503, content={
                    "error": "no LLM provider configured",
                    "detail": "Set PRISM_LLM_PROVIDER + PRISM_LLM_BASE_URL (ollama/lmstudio/openai/mistral/gemini/llamacpp/custom)."})
            instruction = (body.get("instruction") if isinstance(body, dict) else None) or "cleanup"
            target_lang = (body.get("target_lang") if isinstance(body, dict) else None) or "en"
            out_text = process_with_external_llm(
                text=text or "", instruction=instruction,
                target_lang=target_lang, operation=instruction)
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model or cfg.get("model", "external-llm"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": out_text},
                    "finish_reason": "stop",
                }],
            }
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)[:300]})
    _us()._release_sd_best_effort()
    try:
        out = caption_image_b64(image, text or "Describe the image in detail.",
                                max_tokens or 256)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    except EngineMissing as e:
        return JSONResponse(status_code=503,
                            content={"error": "will not load", "detail": str(e)[:200]})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": out["text"]},
            "finish_reason": "stop",
        }],
    }


# ---------- audio ----------

# ---------- batch queue ----------
# Run N jobs sequentially in a background thread (one heavy engine at a time)
# and expose their progress through /v1/batch/{id}. Each job reuses the same
# pipeline as the single endpoint, so nothing is duplicated.
_BATCHES: dict = {}
_BATCHES_LOCK = threading.Lock()


def _batch_start(bid: str, jobs: list) -> None:
    with _BATCHES_LOCK:
        _BATCHES[bid] = {
            "id": bid, "state": "running", "total": len(jobs), "done": 0,
            "percent": 0, "results": [], "error": "", "started": time.time(),
            "cancel": False,
        }


def _batch_update(bid: str, done: int, message: str = "") -> None:
    with _BATCHES_LOCK:
        job = _BATCHES.get(bid)
        if not job:
            return
        job["done"] = done
        job["percent"] = int(100 * done / max(1, job["total"]))
        job["message"] = message


def _batch_finish(bid: str, state: str, error: str = "") -> None:
    with _BATCHES_LOCK:
        job = _BATCHES.get(bid)
        if not job:
            return
        job["state"] = state
        job["error"] = error[:300]
        job["percent"] = 100 if state == "done" else job["percent"]
        job["finished"] = time.time()


def _batch_cancelled(bid: str) -> bool:
    with _BATCHES_LOCK:
        return bool(_BATCHES.get(bid, {}).get("cancel"))


def _run_batch(bid: str, jobs: list, stop_on_error: bool) -> None:
    results = []
    try:
        for i, job in enumerate(jobs):
            if _batch_cancelled(bid):
                _batch_finish(bid, "cancelled", "cancelled by user")
                with _BATCHES_LOCK:
                    _BATCHES[bid]["results"] = results
                return
            kind = (job.get("type") or job.get("kind") or "").lower()
            try:
                if kind == "generate":
                    model_id = _resolve_image_model(job.get("model", ""))
                    _us()._manager.ensure("image", model_id)
                    _us()._release_vlm_best_effort()
                    prompt = job.get("prompt", "")
                    size = job.get("size", "512x512")
                    quality = job.get("quality", "medium")
                    b64 = _try_real_generation(model_id, prompt, size, quality,
                                                job.get("negative_prompt", ""),
                                                job.get("guidance", 1.0),
                                                int(job.get("seed", 0) or time.time()))
                    if not b64:
                        b64 = _placeholder_png_b64()
                        engine = "placeholder"
                    else:
                        engine = "diffusers"
                    meta = _store_b64("image", ".png", b64, bid)
                    results.append({"index": i, "type": "generate", "ok": True,
                                    "size": size, "engine": engine,
                                    "outputs": [meta] if meta else []})
                elif kind == "upscale":
                    image = job.get("image", "")
                    scale = int(job.get("scale", 2))
                    up_model = job.get("model") or "RealESRGAN_x4v3"
                    _us()._manager.ensure("upscale", up_model)
                    out = upscale_image_b64(
                        image, scale, up_model,
                        prompt=job.get("prompt") or "",
                        strength=float(job.get("strength", 0.33)),
                        seed=int(job.get("seed", 0) or 0))
                    meta = _store_b64("image", ".png", out["image"], bid)
                    out["outputs"] = [meta] if meta else []
                    results.append({"index": i, "type": "upscale", "ok": True, **out})
                elif kind == "transcribe":
                    wanted = job.get("model", "turbo") or "turbo"
                    if wanted not in ALLOWED_WHISPER:
                        wanted = "turbo"
                    text, lang, segs = _transcribe_with_whisper(
                        job.get("_path", ""), wanted, job.get("language") or "sk")
                    meta = _store_output("text", ".txt", text.encode("utf-8"), bid)
                    results.append({"index": i, "type": "transcribe", "ok": True,
                                    "text": text, "language": lang,
                                    "segments": len(segs), "outputs": [meta]})
                elif kind == "convert-image":
                    # image bytes passed as base64 under job["image_b64"]
                    import base64 as _b64
                    import tempfile as _tf
                    from pathlib import Path
                    raw = _b64.b64decode(job.get("image_b64", ""))
                    suffix = _P(job.get("filename", "image.png")).suffix or ".png"
                    with _tf.NamedTemporaryFile(delete=False, suffix=suffix) as t:
                        t.write(raw)
                        tmp = t.name
                    try:
                        out_path, _info = convert_image(_P(tmp), preset=job.get("preset", "jpg-95"))
                        out_data = _P(out_path).read_bytes()
                        out_b64 = base64.b64encode(out_data).decode()
                        meta = _store_output("image", _P(out_path).suffix.lower() or ".jpg",
                                             out_data, bid)
                        results.append({"index": i, "type": "convert-image", "ok": True,
                                        "file": out_b64, "outputs": [meta]})
                    finally:
                        try:
                            _P(tmp).unlink(missing_ok=True)
                        except Exception:
                            pass
                else:
                    raise ValueError(f"unknown job type: {kind}")
            except Exception as e:
                logger.warning("batch %s job %d failed: %s", bid, i, str(e)[:200])
                results.append({"index": i, "type": kind, "ok": False,
                                "error": str(e)[:200]})
                if stop_on_error:
                    _batch_finish(bid, "error", str(e))
                    with _BATCHES_LOCK:
                        _BATCHES[bid]["results"] = results
                    return
            _batch_update(bid, i + 1, f"job {i + 1}/{len(jobs)}")
        _batch_finish(bid, "done")
        with _BATCHES_LOCK:
            _BATCHES[bid]["results"] = results
    except Exception as e:
        _batch_finish(bid, "error", str(e))
        with _BATCHES_LOCK:
            _BATCHES[bid]["results"] = results


class BatchRequest(BaseModel):
    jobs: list[dict] = Field(min_length=1)
    stop_on_error: bool = True
    request_id: str | None = ""


@router.post("/v1/batch")
def batch(req: BatchRequest):
    if not req.jobs:
        return JSONResponse(status_code=400, content={"error": "no jobs"})
    bid = (req.request_id or "").strip() or f"batch-{uuid.uuid4().hex[:8]}"
    _batch_start(bid, req.jobs)
    threading.Thread(target=_run_batch, args=(bid, req.jobs, req.stop_on_error),
                     daemon=True).start()
    return {"id": bid, "state": "running", "total": len(req.jobs),
            "request_id": bid}


@router.get("/v1/batch/{bid}")
def batch_status(bid: str):
    with _BATCHES_LOCK:
        job = _BATCHES.get(bid)
        if not job:
            return JSONResponse(status_code=404, content={"error": "unknown batch"})
        snap = dict(job)
    snap["elapsed"] = round(time.time() - snap["started"], 1)
    return snap


@router.post("/v1/batch/{bid}/cancel")
def batch_cancel(bid: str):
    with _BATCHES_LOCK:
        job = _BATCHES.get(bid)
        if not job:
            return JSONResponse(status_code=404, content={"error": "unknown batch"})
        job["cancel"] = True
    return {"id": bid, "state": "cancelling", "cancelled": True}


# ---------- job progress / cancellation ----------
# Long-running work (transcription today, more kinds later) registers here so a
# client can poll /v1/jobs/{id} and stop it with /v1/jobs/{id}/cancel.
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_JOB_HISTORY = 50


class JobCancelled(Exception):
    pass


def _job_start(rid: str, kind: str, total: float = 0.0, message: str = "") -> None:
    with _JOBS_LOCK:
        _JOBS[rid] = {
            "id": rid, "kind": kind, "state": "running", "percent": 0,
            "done": 0.0, "total": float(total or 0), "message": message,
            "started": time.time(), "cancel": False, "error": "",
        }
        if len(_JOBS) > _JOB_HISTORY:
            for old in sorted(_JOBS, key=lambda k: _JOBS[k]["started"])[:10]:
                if old != rid and _JOBS[old]["state"] != "running":
                    _JOBS.pop(old, None)


def _job_update(rid: str, done: float | None = None, total: float | None = None,
                message: str | None = None) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(rid)
        if not job:
            return
        if done is not None:
            job["done"] = float(done)
        if total is not None:
            job["total"] = float(total)
        if message is not None:
            job["message"] = message
        total = job["total"]
        if total > 0:
            job["percent"] = int(max(0, min(100, 100.0 * job["done"] / total)))


def _job_finish(rid: str, state: str = "done", error: str = "") -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(rid)
        if not job:
            return
        job["state"] = state
        job["error"] = error[:300]
        job["percent"] = 100 if state == "done" else job.get("percent", 0)
        job["finished"] = time.time()


def _job_cancelled(rid: str) -> bool:
    with _JOBS_LOCK:
        job = _JOBS.get(rid)
        return bool(job and job.get("cancel"))


def _job_snapshot(rid: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(rid)
        if not job:
            return {"id": rid, "state": ActionType.UNKNOWN.value, "percent": 0}
        snap = dict(job)
    if snap.get("started"):
        end = snap.get("finished") or time.time()
        snap["elapsed"] = round(end - snap["started"], 1)
    return snap


# ---------- job output store ----------
# Every produced artifact is written to pc/output/<kind>/ so a job leaves a file
# behind instead of only a base64 blob in the response.
OUTPUT_KINDS = ("image", "video", "audio", "text")


def _store_output(kind: str, ext: str, data: bytes, rid: str = "", stem: str = "") -> dict:
    """Write one artifact under output/<kind>/ and return its descriptor."""
    kind = kind if kind in OUTPUT_KINDS else "text"
    ext = ext if ext.startswith(".") else f".{ext}"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_rid = "".join(c for c in (rid or "") if c.isalnum() or c in "-_")[:32].strip("-_")
    base = stem or f"{stamp}-{safe_rid or uuid.uuid4().hex[:8]}"
    target = _us().OUTPUT_DIR / kind
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{base}{ext}"
    n = 1
    while path.exists():
        path = target / f"{base}-{n}{ext}"
        n += 1
    path.write_bytes(data)
    return {
        "name": path.name,
        "kind": kind,
        "size": len(data),
        "path": f"{kind}/{path.name}",
        "url": f"/v1/outputs/{kind}/{path.name}",
    }


def _store_b64(kind: str, ext: str, b64: str, rid: str = "") -> dict:
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return {}
    return _store_output(kind, ext, raw, rid)


ALLOWED_AUDIO_EXT = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav",
                     ".webm", ".ogg", ".flac", ".opus"}
MAX_FILE_SIZE = 500 * 1024 * 1024

# CTranslate2 checkpoints whose folder name is not faster-whisper-<id>.
CUSTOM_WHISPER_DIRS = {"ivrit-whisper-turbo": "ivrit-whisper-turbo"}

_WHISPER_CACHE = {}

# OpenAI realtime transcription WebSocket clients. Tracked separately from
# inference so an idle client does not keep the resident model loaded.
_REALTIME_WS_CONNECTIONS = 0


_AV_PATCHED = False


def _patch_av_compat():
    """PyAV 19 removed av.open(metadata_errors=...), which faster-whisper
    still passes. Drop the keyword instead of failing the request."""
    global _AV_PATCHED
    if _AV_PATCHED:
        return
    _AV_PATCHED = True
    try:
        import av
        orig = av.open
        def _open(*args, **kwargs):
            kwargs.pop("metadata_errors", None)
            return orig(*args, **kwargs)
        av.open = _open
    except Exception:
        pass


def _get_whisper_model(source, device: str, compute: str):
    """One loaded faster-whisper model per (source, device, compute)."""
    key = (str(source), device, compute)
    m = _WHISPER_CACHE.get(key)
    if m is None:
        _patch_av_compat()
        from faster_whisper import WhisperModel
        m = WhisperModel(str(source), device=device, compute_type=compute)
        _WHISPER_CACHE[key] = m
    return m


def release_whisper_cache() -> bool:
    had = bool(_WHISPER_CACHE)
    _WHISPER_CACHE.clear()
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    return had


def _transcribe_with_whisper(tmp_path: str, wanted: str, language, rid: str = "",
                             initial_prompt: str = ""):
    # CTranslate2 4.x needs the CUDA 12 runtime (cublas64_12.dll/cudnn64_9.dll).
    # Default to CPU int8 so transcription works without those DLLs; set
    # WHISPER_DEVICE=cuda when the CUDA 12 libraries are installed.
    device = (os.environ.get("WHISPER_DEVICE") or "cpu").strip().lower()
    if device not in ("cpu", "cuda"):
        device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    # Prefer a locally downloaded CTranslate2 model (offline, no HF CDN):
    #   pc/models/faster-whisper-<name>/model.bin
    # whisper_installed() checks the exact expected size + a valid config.json,
    # so a half-finished download is never loaded as if it were complete.
    folder = CUSTOM_WHISPER_DIRS.get(wanted) or f"faster-whisper-{wanted}"
    local = Path(__file__).resolve().parent.parent / "models" / folder
    local_bin = local / "model.bin"
    if wanted in CUSTOM_WHISPER_DIRS:
        use_local = local_bin.is_file() and local_bin.stat().st_size > 50 * 1024 * 1024
    else:
        try:
            from unified.model_catalog import whisper_installed
            use_local = whisper_installed(wanted)
        except Exception:
            use_local = local_bin.is_file() and local_bin.stat().st_size > 50 * 1024 * 1024
    if not use_local:
        local_bin = local / "model.bin"
        if local_bin.is_file():
            logger.warning(
                f"whisper '{wanted}' is not fully downloaded "
                f"({local_bin.stat().st_size} B); will use the HF download")
    source = str(local) if use_local else wanted

    def _run(model, rid: str = "", duration: float = 0.0) -> tuple[str, str, list]:
        kwargs = {"language": language, "beam_size": 1, "vad_filter": True}
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        segs, info = model.transcribe(tmp_path, **kwargs)
        items = []
        for s in segs:
            if rid and _job_cancelled(rid):
                raise JobCancelled()
            txt = (s.text or "").strip()
            if not txt:
                continue
            if rid:
                _job_update(rid, done=float(s.end),
                            message=f"segment {len(items) + 1}")
            items.append({"start": round(float(s.start), 2),
                          "end": round(float(s.end), 2),
                          "text": txt})
        text = " ".join(i["text"] for i in items).strip()
        return text, getattr(info, "language", language), items

    try:
        return _run(_get_whisper_model(source, device, compute), rid)
    except JobCancelled:
        raise
    except Exception as e:
        if device == "cuda":
            logger.warning(f"whisper CUDA failed ({str(e)[:140]}); retrying on CPU int8")
            return _run(_get_whisper_model(source, "cpu", "int8"), rid)
        raise


def _segments_to_srt(segments: list) -> str:
    def ts(t: float) -> str:
        ms = round(t * 1000)
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        sec, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{ts(seg['start'])} --> {ts(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _segments_to_vtt(segments: list) -> str:
    def ts(t: float) -> str:
        ms = round(t * 1000)
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        sec, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"

    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{ts(seg['start'])} --> {ts(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


_REALTIME_LOCK = threading.Lock()


def _resample_pcm16(raw: bytes, rate: int) -> np.ndarray:
    """PCM16 mono at `rate` -> float32 mono at 16 kHz."""
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if rate and rate != 16000 and a.size > 1:
        n = max(1, round(a.size * 16000.0 / rate))
        a = np.interp(np.linspace(0, a.size - 1, n),
                      np.arange(a.size), a).astype(np.float32)
    return a


def _transcribe_pcm(raw: bytes, rate: int, wanted: str, language):
    """Realtime path: in-memory PCM instead of a temp file."""
    device = (os.environ.get("WHISPER_DEVICE") or "cpu").strip().lower()
    if device not in ("cpu", "cuda"):
        device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    folder = CUSTOM_WHISPER_DIRS.get(wanted) or f"faster-whisper-{wanted}"
    local = Path(__file__).resolve().parent.parent / "models" / folder
    try:
        from unified.model_catalog import whisper_installed
        use_local = (wanted in CUSTOM_WHISPER_DIRS
                     and (local / "model.bin").is_file()) or whisper_installed(wanted)
    except Exception:
        use_local = (local / "model.bin").is_file()
    source = str(local) if use_local else wanted
    audio = _resample_pcm16(raw, rate)
    if audio.size == 0:
        return "", language
    with _REALTIME_LOCK:
        model = _get_whisper_model(source, device, compute)
        segs, info = model.transcribe(audio, language=language or None,
                                      beam_size=1, vad_filter=True,
                                      condition_on_previous_text=False)
        text = " ".join((s.text or "").strip() for s in segs).strip()
    return text, getattr(info, "language", language)


def _transcribe_chunked(audio_path: Path, wanted: str, language, rid: str = ""):
    """Long file -> 5-minute chunks: gives real progress and a working cancel.

    faster-whisper decodes the whole file before yielding the first segment, so
    a single pass cannot report progress or be interrupted. Chunking with the
    tail of the previous text as `initial_prompt` keeps the context.
    """
    try:
        dur = float(get_audio_duration(audio_path) or 0.0)
    except Exception:
        dur = 0.0
    if dur <= 600 or _job_cancelled(rid):
        if rid and _job_cancelled(rid):
            raise JobCancelled()
        return _transcribe_with_whisper(str(audio_path), wanted, language, rid)

    from unified.audio_splitter import cleanup_chunks, split_audio_with_ffmpeg
    chunks = split_audio_with_ffmpeg(audio_path, chunk_seconds=300, overlap_seconds=0,
                                     output_dir=Path(tempfile.gettempdir()))
    _job_update(rid, total=dur, message=f"{len(chunks)} chunks")
    texts, segments, prev = [], [], ""
    try:
        for idx, (chunk_path, start, end) in enumerate(chunks):
            if _job_cancelled(rid):
                raise JobCancelled()
            text, _lang, segs = _transcribe_with_whisper(
                str(chunk_path), wanted, language, "", initial_prompt=prev[-300:])
            for seg in segs:
                segments.append({"start": round(start + seg["start"], 2),
                                 "end": round(min(start + seg["end"], end), 2),
                                 "text": seg["text"]})
            texts.append(text)
            prev = (prev + " " + text).strip()[-400:]
            _job_update(rid, done=end, message=f"chunk {idx + 1}/{len(chunks)}")
    finally:
        cleanup_chunks(chunks)
    return " ".join(texts).strip(), language, segments


async def _transcribe_with_splitting(audio_path: Path, wanted: str, language,
                                    rid: str = ""):
    """Transcribe long audio.

    faster-whisper handles long files with its own VAD, so splitting is only
    used for the remote WhisperApi fallback (and for legacy engines).
    """
    _transcribe_single = lambda path, lang: _transcribe_with_whisper(str(path), wanted, lang, rid)

    try:
        import faster_whisper  # noqa: F401
        return _transcribe_single(audio_path, language)
    except ImportError:
        pass

    try:
        # Use the audio splitter utility (remote service path): text only.
        from unified.audio_splitter import transcribe_with_splitting
        text, lang = await transcribe_with_splitting(audio_path, _transcribe_single, language)
        return text, lang, []
    except Exception as e:
        logger.warning(f"Audio splitting failed, falling back to single transcribe: {e}")
        return _transcribe_single(audio_path, language)


async def _transcribe_with_vad(audio_path: Path, wanted: str, language,
                               rid: str = "") -> tuple[str, str]:
    """Transcribe only Silero-detected speech segments.

    Returns ("", language) when the VAD model is not installed so the caller
    can fall back to full-audio transcription without raising.
    """
    from unified import silero_vad

    if not silero_vad.is_installed():
        logger.info("Silero VAD model not installed; processing full audio")
        return "", language or "unknown"

    def _transcribe_segment(segment_path: str, lang):
        text, out_lang, _ = _transcribe_with_whisper(
            str(segment_path), wanted, lang, rid)
        return text, out_lang

    text, out_lang, _segments = await asyncio.to_thread(
        silero_vad.transcribe_with_vad, str(audio_path), language,
        _transcribe_segment)
    return text, out_lang


@router.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(default=None),
    url: str = Form(default=""),
    model: str = Form(default="whisper-1"),
    device: str = Form(default=""),
    language: str = Form(default=None),
    response_format: str = Form(default="json"),
    auto_split: bool = Form(default=True),
    request_id: str = Form(default=""),
    post_process: str = Form(default=""),
    post_process_lang: str = Form(default=""),
    action_extract: str = Form(default=""),
    ai_provider: str = Form(default=""),
    ai_model: str = Form(default=""),
    # --- New OCR/transcription options ---
    output_format: str = Form(default="txt"),       # txt, md, srt, json
    max_chars_per_chunk: int = Form(default=8000),  # text chunk size for AI context limits
    keep_segments: bool = Form(default=False),      # keep segments separate vs merge into one
    use_vad: bool = Form(default=False),            # use Silero VAD to pre-filter speech segments
    post_process_operation: str = Form(default=""), # summary, cleanup, grammar, list, translate, rewrite, meeting_notes
    ):  # End of parameter list
    # Called directly (translations) FastAPI defaults are still Form objects.
    if not isinstance(request_id, str):
        request_id = ""
    rid = request_id or f"req-{uuid.uuid4().hex[:8]}"
    # Normalise values that may arrive as FastAPI Form objects when this
    # function is called directly (e.g. from /transcribe) instead of via the
    # HTTP route, where FastAPI resolves them to plain strings.
    if not isinstance(url, str):
        url = ""
    if not isinstance(device, str):
        device = ""
    if not isinstance(model, str):
        model = "whisper-1"
    if not isinstance(response_format, str):
        response_format = "json"
    if not isinstance(output_format, str):
        output_format = "txt"
    if not isinstance(post_process, str):
        post_process = ""
    if not isinstance(post_process_lang, str):
        post_process_lang = ""
    if not isinstance(post_process_operation, str):
        post_process_operation = ""
    if not isinstance(action_extract, str):
        action_extract = ""
    if not isinstance(ai_provider, str):
        ai_provider = ""
    if not isinstance(ai_model, str):
        ai_model = ""
    up = file
    url = (url or "").strip()
    if url:
        if up is not None:
            return JSONResponse(status_code=400, content={"error": "send either file or url, not both"})
        try:
            from unified.ytdlp import DownloadFailed, YtDlpMissing, download_audio
            tmp_wav = download_audio(url)
        except YtDlpMissing as e:
            return JSONResponse(status_code=503, content={"error": "yt-dlp missing",
                                                         "detail": str(e)[:200]})
        except DownloadFailed as e:
            return JSONResponse(status_code=400, content={"error": "download failed",
                                                         "detail": str(e)[:200]})
        content = Path(tmp_wav).read_bytes()
        Path(tmp_wav).name
        suffix = ".wav"
        up = None
    if up is None and not url:
        return JSONResponse(status_code=400, content={"error": "missing file or url"})
    raw_model = (model or "").strip() or "whisper-1"
    # One resolution for the optional engines: an explicit `nemotron-*` /
    # `sherpa-*` name wins, otherwise `whisper-1` follows PRISM_ASR_ENGINE.
    asr_engine, wanted = resolve_engine(raw_model)
    if asr_engine is None and wanted not in ALLOWED_WHISPER:
        return JSONResponse(status_code=400,
                            content={"error": f"Unknown model '{raw_model}'"})
    if device and device.strip().lower() not in ("cuda", "cpu", ""):
        return JSONResponse(status_code=400, content={"error": "Unknown device"})
    # Validate file extension - support audio and video
    suffix = os.path.splitext(up.filename or "")[1].lower() or ".wav"
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".mpeg", ".mpg", ".3gp", ".ts"}
    AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".aac", ".wma"}
    all_exts = VIDEO_EXTS | AUDIO_EXTS
    if suffix not in all_exts:
        return JSONResponse(status_code=415, content={"error": f"Unsupported format: {suffix}. Supported: {', '.join(sorted(all_exts))}"})
    content = await up.read() if up is not None else content
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file uploaded"})
    if len(content) > MAX_FILE_SIZE:
        return JSONResponse(status_code=413, content={"error": "File too large"})
    # Determine which engine handles this request
    try:
        _us()._manager.ensure("audio", wanted)
    except ModelLoadingError as e:
        return _us()._raise_loading(e)
    _job_start(rid, "transcribe", 0.0, f"model={wanted}")
    logger.info(f"transcribe: {len(content)} bytes, model={wanted}, lang={language}, format={response_format}, output_format={output_format}")
    import tempfile
    tmp = None
    segments = []
    duration_hint = 0.0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)
        try:
            duration_hint = float(get_audio_duration(Path(tmp)) or 0.0)
            _job_update(rid, total=duration_hint)
        except Exception:
            pass
        if asr_engine is not None:
            # An explicit engine choice must never fall through to the
            # WhisperApi proxy: that would silently return another engine's
            # output.
            if not asr_engine.installed(wanted):
                _job_finish(rid, "error", f"{wanted} not installed")
                return JSONResponse(status_code=503, content={
                    "error": f"{wanted} not installed",
                    "detail": "install it from the Models tab, or run "
                              + asr_engine.install_hint})
            try:
                logger.info("transcribe: engine=%s model=%s lang=%s",
                            asr_engine.name, wanted, language)
                text, lang = await asyncio.to_thread(
                    asr_engine.transcribe, tmp, language, wanted)
                segments = []
                engine = asr_engine.name
            except JobCancelled:
                raise
            except Exception as e:
                _job_finish(rid, "error", str(e)[:200])
                logger.error("%s transcribe failed: %s",
                             asr_engine.name, str(e)[:400])
                return JSONResponse(status_code=500, content={
                    "error": f"{asr_engine.name} failed", "detail": str(e)[:300]})
        else:
            try:
                split_enabled = auto_split
                logger.info(f"transcribe: starting engine (split={split_enabled})")
                text = ""
                if use_vad and not split_enabled:
                    # Silero VAD pre-filter: transcribe only detected speech.
                    # Long files stay on the splitting path, which already
                    # chunks audio and reports progress.
                    try:
                        text, lang = await _transcribe_with_vad(
                            Path(tmp), wanted, language, rid)
                        if text:
                            segments = []
                            engine = "whisper+silero_vad"
                    except Exception as e:
                        logger.warning(
                            "Silero VAD failed, falling back to full transcription: %s",
                            str(e)[:200])
                if not text:
                    if split_enabled:
                        text, lang, segments = await _transcribe_with_splitting(
                            Path(tmp), wanted, language, rid)
                    else:
                        text, lang, segments = await asyncio.to_thread(
                            _transcribe_chunked, Path(tmp), wanted, language, rid)
                    engine = "local"
            except JobCancelled:
                # Must not fall through to the proxy: the user asked to stop.
                raise
            except Exception as e:
                logger.warning("local whisper missing (%s), proxy %s",
                                 str(e)[:120], _us().WHISPER_SERVICE_URL)
                proxied = transcribe_via_service(
                    _us().WHISPER_SERVICE_URL, content, up.filename or "audio.wav",
                    wanted, language)
                if proxied is not None:
                    text, lang = proxied
                    segments = []
                    engine = "whisper-service"
                    # nothing is kept in RAM locally → release the gate immediately for images
                    _us()._manager.unload("audio-proxied")
                else:
                    text, lang = "[unified stub — start the WhisperApi service or install faster-whisper]", language
                    segments = []
                    engine = "stub"
        # External LLM post-processing via user-configured provider (Ollama, LM Studio,
        # public API, llama.cpp, etc.). The PC server does NOT download or manage any
        # text-generation model — that is the user's responsibility via their own LLM loader.
        if post_process and text and not text.startswith("[unified stub"):
            try:
                from unified.llm_providers import process_with_external_llm
                pp_lang = post_process_lang or lang or "sk"
                # The provider gate can block for a long time waiting on a
                # local model slot; keep it off the event loop.
                text = await asyncio.to_thread(
                    process_with_external_llm, text,
                    instruction=post_process,
                    target_lang=pp_lang,
                    operation=post_process_operation or None,
                )
                if text and not text.startswith("[external llm"):
                    engine = engine + "+external-llm"
                    logger.info("External LLM post-processing applied: %s", post_process)
            except Exception as e:
                logger.warning("External LLM post-processing failed: %s", str(e)[:200])

        # Optional AI action extraction (calendar, todo, email, reminder, etc.)
        if action_extract and text and not text.startswith("[unified stub"):
            try:
                ai_client = get_ai_client()
                extractor = ActionExtractor(ai_client)
                actions = await extractor.extract(text, provider=ai_provider, model=ai_model)
                action_results = await asyncio.to_thread(process_actions, actions)
                logger.info("Action extraction: %d action(s) extracted from transcription", len(actions))
            except Exception as e:
                logger.warning("Action extraction failed: %s", str(e)[:200])
                action_results = []
        else:
            action_results = []
        _job_finish(rid, "done")
        fmt = (response_format or "json").strip().lower()
        output_format_param = (output_format or "txt").strip().lower()
        if output_format_param not in ("txt", "md", "srt", "json"):
            output_format_param = "txt"
        # OpenAI-compatible clients send response_format=json/text; honour it
        # when the Prism-specific output_format was left at its default.
        if (output_format or "txt").strip().lower() == "txt":
            if fmt in ("json", "verbose_json"):
                output_format_param = "json"
            elif fmt == "text":
                output_format_param = "txt"
        # Build segments list from text if no segments (e.g., whisper-service)
        if not segments and text:
            segments = [{"text": text, "start": 0.0, "end": duration_hint or 0.0}]
        # Format output
        if output_format_param == "json":
            resp_data = {
                "text": text,
                "language": lang,
                "engine": engine,
                "duration": get_audio_duration(Path(tmp)) if tmp else 0.0,
                "segments": segments,
                "outputs": [{"name": f"{rid}.{output_format_param}", "size": len(text.encode("utf-8"))}],
                "actions": [a.to_dict() for a in action_results] if action_results else [],
            }
            if fmt == "verbose_json":
                meta = _store_output("text", f".{output_format_param}", text.encode("utf-8"), rid)
                resp_data["outputs"] = [meta]
            return resp_data
        elif output_format_param == "txt":
            formatted = format_transcription(segments, "txt", language or "unknown")
            meta = _store_output("text", ".txt", formatted.encode("utf-8"), rid)
            from fastapi.responses import PlainTextResponse
            resp = PlainTextResponse(formatted)
            resp.headers["X-Output-Path"] = meta["path"]
            resp.headers["X-Engine"] = engine
            resp.headers["X-Language"] = language or "unknown"
            return resp
        elif output_format_param == "md":
            formatted = format_transcription(segments, "md", language or "unknown")
            meta = _store_output("text", ".md", formatted.encode("utf-8"), rid)
            from fastapi.responses import PlainTextResponse
            resp = PlainTextResponse(formatted)
            resp.headers["X-Output-Path"] = meta["path"]
            resp.headers["X-Engine"] = engine
            resp.headers["X-Language"] = language or "unknown"
            return resp
        elif output_format_param == "srt":
            formatted = format_transcription(segments, "srt", language or "unknown")
            meta = _store_output("text", ".srt", formatted.encode("utf-8"), rid)
            from fastapi.responses import PlainTextResponse
            resp = PlainTextResponse(formatted)
            resp.headers["X-Output-Path"] = meta["path"]
            resp.headers["X-Engine"] = engine
            resp.headers["X-Language"] = language or "unknown"
            return resp
        else:  # json default
            resp_data = {
                "text": text,
                "language": lang,
                "engine": engine,
                "duration": get_audio_duration(Path(tmp)) if tmp else 0.0,
                "segments": segments,
                "outputs": [{"name": f"{rid}.json", "size": len(text.encode("utf-8"))}],
                "actions": [a.to_dict() for a in action_results] if action_results else [],
            }
            if fmt == "verbose_json":
                meta = _store_output("text", ".json", text.encode("utf-8"), rid)
                resp_data["outputs"] = [meta]
            return resp_data
    except JobCancelled:
        _job_finish(rid, "cancelled")
        return JSONResponse(status_code=409, content={
            "error": "cancelled", "request_id": rid})
    except Exception as e:
        _job_finish(rid, "error", str(e))
        raise
    finally:
        try:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/v1/audio/transcribe-batch")
async def transcribe_batch(
    files: list[UploadFile] = File(default=None),
    model: str = Form(default="whisper-1"),
    device: str = Form(default=""),
    language: str = Form(default=None),
    response_format: str = Form(default="json"),
    output_format: str = Form(default="txt"),
    auto_split: bool = Form(default=True),
    max_chars_per_chunk: int = Form(default=8000),
    keep_segments: bool = Form(default=False),
    request_id: str = Form(default=""),
    post_process: str = Form(default=""),
    post_process_lang: str = Form(default=""),
    action_extract: str = Form(default=""),
    ai_provider: str = Form(default=""),
    ai_model: str = Form(default=""),
    post_process_operation: str = Form(default=""), # summary, cleanup, grammar, list, translate, rewrite, meeting_notes
):
    """Batch transcription endpoint for multiple audio/video files.

    Transcribes multiple files in a single request. Returns results for each
    file individually plus a combined result if keep_segments=False.

    Supported input formats: .wav, .mp3, .m4a, .flac, .ogg, .webm, .aac, .wma,
    .mp4, .avi, .mov, .mkv, .flv, .wmv, .mpeg, .mpg, .3gp, .ts

    Output formats: txt, md, srt, json
    """
    if not files or all(f.filename == "" for f in files):
        return JSONResponse(status_code=400, content={"error": "No files provided"})

    rid = request_id or f"req-{uuid.uuid4().hex[:8]}"
    results = []

    raw_model = (model or "").strip() or "whisper-1"
    asr_engine, wanted = resolve_engine(raw_model)
    if asr_engine is None and wanted not in ALLOWED_WHISPER:
        return JSONResponse(status_code=400,
                            content={"error": f"Unknown model '{raw_model}'"})

    for file in files:
        if not file.filename:
            continue
        t_start = time.perf_counter()
        try:
            content = await file.read()
            if not content:
                results.append({
                    "name": file.filename,
                    "status": "error",
                    "error": "Empty file",
                })
                continue

            if len(content) > MAX_FILE_SIZE:
                results.append({
                    "name": file.filename,
                    "status": "error",
                    "error": "File too large",
                })
                continue

            suffix = os.path.splitext(file.filename or "")[1].lower() or ".wav"
            VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".mpeg", ".mpg", ".3gp", ".ts"}
            AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".aac", ".wma"}
            all_exts = VIDEO_EXTS | AUDIO_EXTS
            if suffix not in all_exts:
                results.append({
                    "name": file.filename,
                    "status": "error",
                    "error": f"Unsupported format: {suffix}",
                })
                continue

            import tempfile
            tmp_path = None
            segments = []
            duration_hint = 0.0
            try:
                # Save uploaded file to temp
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(content)
                tmp_path = tmp.name
                tmp.close()
                # faster-whisper shells out to ffmpeg for video containers, so
                # video files go straight to the engine without a pre-pass.
                try:
                    duration_hint = float(get_audio_duration(Path(tmp_path)) or 0.0)
                except Exception:
                    duration_hint = 0.0

                if asr_engine is not None:
                    if not asr_engine.installed(wanted):
                        results.append({
                            "name": file.filename,
                            "status": "error",
                            "error": f"{wanted} not installed",
                        })
                        continue
                    text, lang = await asyncio.to_thread(
                        asr_engine.transcribe, tmp_path, language, wanted)
                    engine = asr_engine.name
                else:
                    if auto_split:
                        text, lang, segments = await _transcribe_with_splitting(Path(tmp_path), wanted, language, f"{rid}-{file.filename}")
                    else:
                        text, lang, segments = await asyncio.to_thread(_transcribe_chunked, Path(tmp_path), wanted, language, f"{rid}-{file.filename}")
                    engine = "local"

                if post_process and text:
                    try:
                        from unified.llm_providers import process_with_external_llm
                        pp_lang = post_process_lang or lang or "sk"
                        text = await asyncio.to_thread(
                            process_with_external_llm, text,
                            instruction=post_process,
                            target_lang=pp_lang,
                            operation=post_process_operation or None,
                        )
                        engine = engine + "+external-llm"
                    except Exception as e:
                        logger.warning(f"External LLM post-processing failed: {e}")

                segments_list = segments or (
                    [{"text": text, "start": 0.0, "end": duration_hint or 0.0}]
                    if text else []
                )

                result_text = format_transcription(
                    segments_list, output_format, language or "unknown")

                results.append({
                    "name": file.filename,
                    "status": "success",
                    "text": text,
                    "formatted": result_text,
                    "model": model,
                    "language": lang,
                    "engine": engine,
                    "elapsed": round(time.perf_counter() - t_start, 2),
                    "duration": round(duration_hint, 2),
                    "segments": segments_list,
                    "outputs": [{"name": f"{rid}-{file.filename}.{output_format}", "size": len(result_text.encode("utf-8"))}],
                })

            except Exception as e:
                logger.error(f"Transcription failed for {file.filename}: {e}")
                results.append({
                    "name": file.filename,
                    "status": "error",
                    "error": str(e)[:200],
                })
            finally:
                if tmp_path:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"Batch transcription failed for {file.filename}: {e}")
            results.append({
                "name": file.filename,
                "status": "error",
                "error": str(e)[:200],
            })

    # Build combined result
    combined_text = None
    combined_outputs = []
    successful = [r for r in results if r.get("status") == "success"]
    if successful:
        all_segments = []
        for r in successful:
            for seg in r.get("segments", []):
                s = dict(seg)
                s["file"] = r["name"]
                all_segments.append(s)

        if keep_segments:
            # One labelled block per input file, keeping each file's own result.
            if output_format == "json":
                combined_text = json.dumps(
                    {r["name"]: r.get("segments", []) for r in successful},
                    ensure_ascii=False, indent=1)
            else:
                combined_text = "\n\n".join(
                    f"## {r['name']}\n\n{r['formatted']}" for r in successful)
        else:
            combined_text = format_transcription(
                all_segments, output_format, language or "unknown")

        combined_outputs = [{"name": f"{rid}-combined.{output_format}", "size": len(combined_text.encode("utf-8"))}]

    return {
        "rid": rid,
        "total": len(results),
        "successful": len(successful),
        "failed": len([r for r in results if r.get("status") == "error"]),
        "format": output_format,
        "results": results,
        "combined_text": combined_text,
        "combined_outputs": combined_outputs if combined_outputs else None,
    }


@router.websocket("/v1/realtime")
async def realtime_asr(ws: WebSocket):
    """OpenAI transcription-session WebSocket (Cindy "OpenAI transcription").

    Client: session.update, input_audio_buffer.append (base64 PCM16),
    input_audio_buffer.commit, session.finish.
    Server: session.updated, input_audio_buffer.committed,
    conversation.item.input_audio_transcription.completed, plus failed/error.

    Audio arrives at the rate announced in session.update (Cindy uses 24 kHz);
    it is resampled to 16 kHz and transcribed on commit. No partial deltas:
    Cindy accepts a final transcript per commit.
    """
    await ws.accept()
    global _REALTIME_WS_CONNECTIONS
    _REALTIME_WS_CONNECTIONS += 1
    buf = bytearray()
    rate = 24000
    model_name = "turbo"
    language = None
    try:
        while True:
            try:
                msg = await ws.receive_json()
            except WebSocketDisconnect:
                raise
            except Exception:
                break
            typ = msg.get("type") if isinstance(msg, dict) else None
            if typ == "session.update":
                session = msg.get("session") or {}
                audio = session.get("audio") or {}
                inp = audio.get("input") or {}
                fmt = inp.get("format") or {}
                try:
                    rate = int(fmt.get("rate") or session.get("sample_rate") or rate)
                except (TypeError, ValueError):
                    rate = 24000
                tr = (inp.get("transcription")
                      or session.get("input_audio_transcription") or {})
                model_name = str(tr.get("model") or model_name or "turbo").strip()
                language = tr.get("language") or session.get("language") or language
                await ws.send_json({"type": "session.updated", "session": session})
            elif typ == "input_audio_buffer.append":
                try:
                    buf.extend(base64.b64decode(msg.get("audio") or ""))
                except Exception:
                    pass
            elif typ == "input_audio_buffer.clear":
                buf.clear()
            elif typ == "input_audio_buffer.commit":
                item_id = f"item_{uuid.uuid4().hex[:12]}"
                await ws.send_json({"type": "input_audio_buffer.committed",
                                    "item_id": item_id})
                raw = bytes(buf)
                buf.clear()
                wanted = (model_name or "turbo").strip().lower()
                if wanted in ("gpt-realtime-whisper", "whisper-1",
                              "gpt-4o-transcribe", "gpt-4o-mini-transcribe"):
                    wanted = "turbo"
                if wanted not in ALLOWED_WHISPER:
                    wanted = "turbo"
                try:
                    _us()._manager.begin()
                    _us()._manager.ensure("audio", wanted)
                    text, _lang = await asyncio.to_thread(
                        _transcribe_pcm, raw, rate, wanted, language)
                    await ws.send_json({
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": item_id,
                        "content_index": 0,
                        "transcript": text,
                        "usage": {"type": "duration",
                                  "seconds": round(len(raw) / 2 / max(rate, 1), 2)},
                    })
                except Exception as e:
                    logger.warning("realtime transcription failed: %s", str(e)[:200])
                    await ws.send_json({
                        "type": "conversation.item.input_audio_transcription.failed",
                        "item_id": item_id,
                        "error": {"type": "server_error",
                                  "message": str(e)[:300]},
                    })
                finally:
                    _us()._manager.end()
            elif typ in ("session.finish", "session.close"):
                await ws.send_json({"type": "session.finished"})
                break
            # Unknown control events are ignored on purpose.
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("realtime socket error: %s", str(e)[:200])
        try:
            await ws.send_json({"type": "error", "error": {
                "type": "server_error", "message": str(e)[:300]}})
        except Exception:
            pass
    finally:
        _REALTIME_WS_CONNECTIONS = max(0, _REALTIME_WS_CONNECTIONS - 1)


@router.post("/v1/audio/translations")
async def translations(
    file: UploadFile = File(default=None),
    model: str = Form(default="whisper-1"),
):
    r = await transcriptions(file=file, url="", model=model,
                             device="", language=None,
                             response_format="json", auto_split=True,
                             request_id="")
    if isinstance(r, dict) and "text" in r:
        return {"text": r["text"]}
    return r


@router.post("/v1/images/ocr")
async def ocr_endpoint(
    file: UploadFile = File(default=None),
    language: str = Form(default="auto"),
    model: str = Form(default="auto"),
    task: str = Form(default="ocr"),
):
    """OCR or image description with a selectable model.

    `model` is any installed id from `GET /v1/models/catalog` with kind `ocr`
    (`nemotron-ocr-0`, `granite-docling-258m`, `glm-ocr`,
    `paddleocr-vl-1.6`, `lightonocr-2-1b`, ...), or `auto` (default), which
    prefers the UI-aware GLM-OCR and falls back to Nemotron. `task=ocr` returns
    text; `task=describe` returns an image description and requires
    `qwen3-vl-2b`.
    """
    if file is None:
        return JSONResponse(status_code=400, content={"error": "missing file"})

    from unified.ocr_engine import get_ocr_engine

    engine = get_ocr_engine(model)
    if task == "describe" and not engine.supports_describe:
        return JSONResponse(status_code=400, content={
            "error": "this model does not describe images",
            "detail": "use model=qwen3-vl-2b for task=describe"})
    if not engine.is_installed():
        return JSONResponse(status_code=503, content={
            "error": "OCR model not installed",
            "model": engine.model_id,
            "detail": f"Download {engine.model_id} from the Models tab, or run "
                      f"python pc/download_models.py {engine.model_id}"})

    loaded = await asyncio.to_thread(engine.init)
    if not loaded:
        return JSONResponse(status_code=503, content={
            "error": "OCR model failed to load"})

    try:
        import os
        import tempfile
        # Save uploaded file to temp
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".jpg")[1])
        content = await file.read()
        tmp.write(content)
        tmp.close()

        try:
            text = await asyncio.to_thread(engine.transcribe, tmp.name, language=language, task=task)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

        return {"text": text, "model": engine.model_id,
                "language": language, "task": task}
    except Exception as e:
        logger.error(f"OCR endpoint failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})


@router.post("/v1/documents/extract")
async def document_extract(
    file: UploadFile = File(default=None),
    password: str = Form(default=""),
    language: str = Form(default="auto"),
):
    """Extract text from txt/md/docx/pdf/image files."""
    if file is None:
        return JSONResponse(status_code=400, content={"error": "missing file"})
    try:
        from unified import document_extract as _de
        content = await file.read()
        result = _de.extract_bytes(content, file.filename or "document", password=password, language=language)
        return {"text": result.get("text", ""), "method": result.get("method", ""),
                "pages": result.get("pages", 0), "filename": file.filename}
    except Exception as e:
        logger.error(f"document extract failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})


@router.post("/transcribe")
async def transcribe_alias(
    file: UploadFile = File(...),
    language: str = Form(default=None),
    model: str = Form(default="whisper-1"),
    response_format: str = Form(default="json"),
    auto_split: bool = Form(default=True),
    request_id: str = Form(default=""),
    post_process: str = Form(default=""),
    post_process_lang: str = Form(default=""),
    post_process_operation: str = Form(default=""),
    action_extract: str = Form(default=""),
    ai_provider: str = Form(default=""),
    ai_model: str = Form(default=""),
):
    return await transcriptions(
        file=file,
        url="",
        language=language,
        model=model,
        device="",
        response_format=response_format,
        auto_split=auto_split,
        request_id=request_id,
        post_process=post_process,
        post_process_lang=post_process_lang,
        post_process_operation=post_process_operation,
        action_extract=action_extract,
        ai_provider=ai_provider,
        ai_model=ai_model,
    )


@router.post("/v1/text/process")
async def text_process(
    text: str = Form(default=""),
    instruction: str = Form(default="cleanup"),
    target_lang: str = Form(default="sk"),
    operation: str = Form(default=""),  # summary, cleanup, grammar, list, translate, rewrite, meeting_notes
):
    """Post-process text using an external LLM provider (Ollama, LM Studio, public API, etc.).

    The PC server does NOT download or manage any text-generation model. Configure
    via env vars: PRISM_LLM_PROVIDER, PRISM_LLM_BASE_URL, PRISM_LLM_API_KEY,
    PRISM_LLM_MODEL, PRISM_LLM_ENDPOINT.

    Default prompts per operation (overridable via PRISM_PROMPT_<OPERATION> env var):
      summary, cleanup, grammar, list, translate, rewrite, meeting_notes
    """
    if not text.strip():
        return JSONResponse(status_code=400, content={"error": "missing text"})
    try:
        from unified.llm_providers import process_with_external_llm
        result = process_with_external_llm(
            text, instruction=instruction, target_lang=target_lang,
            operation=operation or None,
        )
        cfg = {
            "provider": os.environ.get("PRISM_LLM_PROVIDER", "none"),
            "model": os.environ.get("PRISM_LLM_MODEL", "gemma4:12b"),
            "base_url": os.environ.get("PRISM_LLM_BASE_URL", ""),
        }
        return {"text": result, "original": text, "instruction": instruction,
                "target_lang": target_lang, "operation": operation,
                "engine": "external-llm", "config": cfg}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})


# ---------- text-to-speech ----------
class SpeechRequest(BaseModel):
    model: str | None = "supertonic-3"
    input: str = Field(min_length=1)
    voice: str | None = "0"
    response_format: str | None = "mp3"
    speed: float | None = Field(default=1.0, strict=True)


@router.post("/v1/audio/speech", responses={
    200: {"content": {"audio/mpeg": {"schema": {"type": "string", "format": "binary"}}},
          "description": "Synthesized speech audio"},
    400: {"description": "Bad request (missing input)"},
    422: {"description": "Validation error"},
})
async def audio_speech(req: SpeechRequest):
    if not (req.input or "").strip():
        return JSONResponse(status_code=400, content={"error": "missing input"})
    model = (req.model or "supertonic-3").strip().lower()
    try:
        if model.startswith("kitten"):
            from unified import kitten_tts as _kt
            if not _kt.is_installed():
                raise RuntimeError("kitten-tts not installed: pip install kittenml")
            data = _kt.synthesize_bytes(
                req.input, voice=req.voice or "Jasper",
                speed=req.speed or 1.0,
                fmt=req.response_format or "wav",
            )
            ctype = "audio/wav" if (req.response_format or "wav").lower() in ("wav", "wave") else "audio/mpeg"
            return Response(content=data, media_type=ctype,
                           headers={"Content-Disposition": 'attachment; filename="speech.' +
                                   (req.response_format or "wav").lower().lstrip(".") + '"'})
        from unified import tts_engine as _tts
        data, ctype = _tts.synthesize_bytes(
            req.input, voice=req.voice or "0",
            speed=req.speed or 1.0,
            fmt=req.response_format or "mp3",
        )
        return Response(content=data, media_type=ctype,
                       headers={"Content-Disposition": 'attachment; filename="speech.' +
                               (req.response_format or "mp3").lower().lstrip(".") + '"'})
    except Exception as e:
        err = str(e)[:300]
        try:
            from unified import tts_engine as _tts2
            err = (_tts2.last_error() or err)[:300]
        except Exception:
            pass
        status = 503 if "not installed" in err.lower() or "model" in err.lower() else 500
        return JSONResponse(status_code=status, content={"error": err})


@router.get("/v1/audio/voices")
def audio_voices():
    try:
        from unified import tts_engine as _tts
        voices = _tts.voices()
        engine = "supertonic-3"
        installed = _tts.is_installed()
    except Exception:
        voices, engine, installed = [], "supertonic-3", False
    try:
        from unified import kitten_tts as _kt
        if _kt.is_installed():
            voices = voices + _kt.voices()
            engine = engine + "+kitten-tts"
            installed = True
    except Exception:
        pass
    return {"voices": voices, "engine": engine, "installed": installed}


# ---------- webui ----------

@router.get("/webui", response_class=HTMLResponse)
def webui():
    for name in ("webui_unified.html", "webui.html"):
        p = HERE / name
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    return ('<h3>Unified Standalone</h3><ul>'
            '<li>POST /v1/images/generations|edits|inpaint|harmonize|upscale</li>'
            '<li>POST /v1/audio/transcriptions|translations</li></ul>')




# ---------- ffmpeg tools ----------

class ConvertRequest(BaseModel):
    preset: str = "mp4-h264"
    resize: str | None = None
    trim_start: float | None = None
    trim_end: float | None = None
    request_id: str | None = ""


class AudioConvertRequest(BaseModel):
    preset: str = "mp3-128"
    trim_start: float | None = None
    trim_end: float | None = None
    request_id: str | None = ""


class TrimRequest(BaseModel):
    start: float = 0.0
    end: float = 10.0
    request_id: str | None = ""


@router.get("/v1/ffmpeg/presets")
def _presets_route():
    return get_presets()


@router.post("/v1/ffmpeg/convert-video")
async def convert_video_endpoint(
    file: UploadFile = File(...),
    preset: str = Form(default="mp4-h264"),
    resize: str = Form(default=""),
    trim_start: str = Form(default=""),
    trim_end: str = Form(default=""),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file"})
    if len(content) > MAX_FILE_SIZE:
        return JSONResponse(status_code=413, content={"error": "File too large"})

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)

        ts = float(trim_start) if trim_start.strip() else None
        te = float(trim_end) if trim_end.strip() else None

        out_path, info = convert_video(
            Path(tmp), preset=preset,
            resize=resize.strip() or None,
            trim_start=ts, trim_end=te
        )

        out_data = Path(out_path).read_bytes()
        out_b64 = base64.b64encode(out_data).decode()
        _ext = Path(out_path).suffix.lower() or ".bin"
        _kind = ("image" if _ext in (".jpg", ".jpeg", ".png", ".webp", ".avif") else
                 "audio" if _ext in (".mp3", ".m4a", ".aac", ".wav", ".opus", ".flac") else
                 "video")
        out_meta = _store_output(_kind, _ext, out_data, rid)

        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "file": out_b64,
            "outputs": [out_meta],
            "filename": Path(out_path).name,
            "elapsed": 0,
            "request_id": rid,
            "info": {
                "duration": info.get("format", {}).get("duration"),
                "size": info.get("format", {}).get("size"),
                "format": info.get("format", {}).get("format_name"),
            }
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})
    finally:
        try:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/v1/ffmpeg/convert-audio")
async def convert_audio_endpoint(
    file: UploadFile = File(...),
    preset: str = Form(default="mp3-128"),
    trim_start: str = Form(default=""),
    trim_end: str = Form(default=""),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)

        ts = float(trim_start) if trim_start.strip() else None
        te = float(trim_end) if trim_end.strip() else None

        out_path, info = convert_audio(
            Path(tmp), preset=preset,
            trim_start=ts, trim_end=te
        )

        out_data = Path(out_path).read_bytes()
        out_b64 = base64.b64encode(out_data).decode()
        _ext = Path(out_path).suffix.lower() or ".bin"
        _kind = ("image" if _ext in (".jpg", ".jpeg", ".png", ".webp", ".avif") else
                 "audio" if _ext in (".mp3", ".m4a", ".aac", ".wav", ".opus", ".flac") else
                 "video")
        out_meta = _store_output(_kind, _ext, out_data, rid)

        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "file": out_b64,
            "outputs": [out_meta],
            "filename": Path(out_path).name,
            "request_id": rid,
            "info": {
                "duration": info.get("format", {}).get("duration"),
                "size": info.get("format", {}).get("size"),
                "format": info.get("format", {}).get("format_name"),
            }
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})
    finally:
        try:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/v1/ffmpeg/convert-image")
async def convert_image_endpoint(
    file: UploadFile = File(...),
    preset: str = Form(default="jpg-95"),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file"})
    suffix = Path(file.filename or "image.png").suffix or ".png"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)
        out_path, info = convert_image(Path(tmp), preset=preset)
        _raw = Path(out_path).read_bytes()
        out_b64 = base64.b64encode(_raw).decode()
        _ext = Path(out_path).suffix.lower() or ".jpg"
        out_meta = _store_output("image", _ext, _raw, rid)
        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "file": out_b64,
            "outputs": [out_meta],
            "filename": Path(out_path).name,
            "request_id": rid,
            "info": {
                "size": info.get("format", {}).get("size"),
                "format": info.get("format", {}).get("format_name"),
            },
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})
    finally:
        try:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


_MEDIA_EXT_KINDS = {
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".webp": "image",
    ".avif": "image",
    ".mp3": "audio", ".m4a": "audio", ".aac": "audio", ".wav": "audio",
    ".opus": "audio", ".flac": "audio",
}


def _media_output_kind(ext: str) -> str:
    return _MEDIA_EXT_KINDS.get(ext, "video")


async def _run_media_transform(file: UploadFile, rid: str, default_name: str,
                               transform, runtime_status: int = 500):
    """Read an upload, run an ffmpeg helper, store and clean up the result.

    The extract/trim/compress endpoints shared the same 40-line temp-file,
    base64, output-store and cleanup block; keep one implementation.
    """
    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    suffix = Path(file.filename or default_name).suffix or ".mp4"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)

        out_path, info = transform(Path(tmp))
        out_data = Path(out_path).read_bytes()
        out_b64 = base64.b64encode(out_data).decode()
        ext = Path(out_path).suffix.lower() or ".bin"
        out_meta = _store_output(_media_output_kind(ext), ext, out_data, rid)

        try:
            Path(out_path).unlink(missing_ok=True)
        except OSError:
            pass

        return {
            "file": out_b64,
            "outputs": [out_meta],
            "filename": Path(out_path).name,
            "request_id": rid,
            "info": {
                "duration": info.get("format", {}).get("duration"),
                "size": info.get("format", {}).get("size"),
            },
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    except RuntimeError as e:
        return JSONResponse(status_code=runtime_status,
                            content={"error": str(e)[:200]})
    finally:
        try:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass


@router.post("/v1/ffmpeg/extract-audio")
async def extract_audio_endpoint(
    file: UploadFile = File(...),
    format: str = Form(default="wav"),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    return await _run_media_transform(
        file, rid, "video.mp4",
        lambda path: extract_audio(path, audio_format=format))


@router.post("/v1/ffmpeg/trim")
async def trim_endpoint(
    file: UploadFile = File(...),
    start: float = Form(default=0.0),
    end: float = Form(default=10.0),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    return await _run_media_transform(
        file, rid, "media.mp4", lambda path: trim_media(path, start, end))


@router.post("/v1/ffmpeg/compress")
async def compress_endpoint(
    file: UploadFile = File(...),
    target_size_mb: float = Form(default=10.0),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    return await _run_media_transform(
        file, rid, "video.mp4",
        lambda path: compress_video(path, target_size_mb),
        runtime_status=400)


@router.post("/v1/ffmpeg/info")
async def media_info_endpoint(
    file: UploadFile = File(...),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    suffix = Path(file.filename or "media.mp4").suffix or ".mp4"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)

        info = get_media_info(Path(tmp))
        return {"info": info, "request_id": rid}
    finally:
        try:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass




@router.post("/v1/ffmpeg/run")
async def ffmpeg_run_endpoint(
    file: UploadFile = File(...),
    flags: str = Form(default=""),
    output_name: str = Form(default="output.mp4"),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    import re as _re
    if _re.search(r"[;&|`$<>]", flags):
        return JSONResponse(status_code=400, content={"error": "invalid characters in flags"})
    toks = flags.split()
    if any(t == "-i" or t.startswith("-i") for t in toks):
        return JSONResponse(status_code=400, content={"error": "'-i' is not allowed in flags"})

    suffix = Path(file.filename or "media.mp4").suffix or ".mp4"
    tmp = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)

        oname = Path(output_name).name or "output.mp4"
        if not Path(oname).suffix:
            oname += suffix
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=Path(oname).suffix).name

        import shutil as _sh2
        ff = _sh2.which("ffmpeg") or "ffmpeg"
        cmd = [ff, "-y", "-i", tmp] + toks + [out_path]
        proc = subprocess.run(cmd, capture_output=True, timeout=600)
        if proc.returncode != 0 or not Path(out_path).exists():
            err = (proc.stderr or b"").decode(errors="replace")[-500:]
            return JSONResponse(status_code=500, content={"error": "ffmpeg failed", "detail": err})

        out_data = Path(out_path).read_bytes()
        out_b64 = base64.b64encode(out_data).decode()
        _ext = Path(out_path).suffix.lower() or ".bin"
        _kind = ("image" if _ext in (".jpg", ".jpeg", ".png", ".webp", ".avif") else
                 "audio" if _ext in (".mp3", ".m4a", ".aac", ".wav", ".opus", ".flac") else
                 "video")
        out_meta = _store_output(_kind, _ext, out_data, rid)

        return {
            "file": out_b64,
            "outputs": [out_meta],
            "filename": oname,
            "elapsed": 0,
            "request_id": rid,
            "engine": "ffmpeg",
        }
    finally:
        for pth in (tmp, out_path):
            try:
                if pth:
                    Path(pth).unlink(missing_ok=True)
            except Exception:
                pass


@router.post("/v1/cleanup")
async def cleanup_endpoint(
    include_cache: bool = Form(default=True),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    deleted = 0
    # Remove all produced outputs.
    try:
        if _us().OUTPUT_DIR.exists():
            for p in _us().OUTPUT_DIR.rglob("*"):
                if p.is_file():
                    try:
                        p.unlink()
                        deleted += 1
                    except Exception:
                        pass
            # Remove empty dirs.
            for d in sorted(_us().OUTPUT_DIR.rglob("*"), reverse=True):
                if d.is_dir():
                    try:
                        d.rmdir()
                    except Exception:
                        pass
    except Exception as e:
        logger.warning("cleanup outputs failed: %s", e)
    # Whisper / model caches (best-effort).
    if include_cache:
        try:
            import gc
            _WHISPER_CACHE.clear()
            gc.collect()
        except Exception:
            pass
    return {"deleted": deleted, "request_id": rid, "engine": "cleanup"}


# ---------- jobs (progress / cancel) ----------

@router.get("/v1/jobs")
def list_jobs():
    with _JOBS_LOCK:
        items = [dict(j) for j in _JOBS.values()]
    items.sort(key=lambda j: j.get("started", 0), reverse=True)
    return {"count": len(items), "jobs": items[:50]}


@router.get("/v1/jobs/{rid}")
def get_job(rid: str):
    snap = _job_snapshot(rid)
    if snap.get("state") == ActionType.UNKNOWN.value:
        return JSONResponse(status_code=404, content={"error": "unknown job"})
    return snap


@router.post("/v1/jobs/{rid}/cancel")
def cancel_job(rid: str):
    with _JOBS_LOCK:
        job = _JOBS.get(rid)
        if not job:
            return JSONResponse(status_code=404, content={"error": "unknown job"})
        if job["state"] != "running":
            return {"id": rid, "state": job["state"], "cancelled": False}
        job["cancel"] = True
    logger.info("job %s: cancel requested", rid)
    return {"id": rid, "state": "cancelling", "cancelled": True}


# ---------- outputs ----------

MAX_OUTPUT_REL_BYTES = 1024
MAX_OUTPUT_SEGMENT_BYTES = 255
@router.get("/v1/outputs")
def list_outputs(limit: int = 200):
    files = []
    if _us().OUTPUT_DIR.is_dir():
        for p in sorted(_us().OUTPUT_DIR.rglob("*"), key=lambda x: x.stat().st_mtime,
                        reverse=True):
            if not p.is_file():
                continue
            rel = p.relative_to(_us().OUTPUT_DIR).as_posix()
            st = p.stat()
            files.append({
                "name": p.name,
                "kind": p.parent.name,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "path": rel,
                "url": f"/v1/outputs/{rel}",
            })
            if len(files) >= max(1, min(limit, 1000)):
                break
    return {"dir": str(_us().OUTPUT_DIR), "count": len(files), "files": files}


@router.get("/v1/outputs/{rel:path}")
def get_output(rel: str):
    try:
        target = _us()._output_path(rel)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    try:
        is_file = target.is_file()
    except OSError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    if not is_file:
        return JSONResponse(status_code=404, content={"error": "not found"})
    try:
        return FileResponse(target, filename=target.name)
    except OSError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})


@router.delete("/v1/outputs/{rel:path}")
def delete_output(rel: str):
    try:
        target = _us()._output_path(rel)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    try:
        is_file = target.is_file()
    except OSError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    if not is_file:
        return JSONResponse(status_code=404, content={"error": "not found"})
    try:
        target.unlink()
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "not found"})
    except OSError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    return {"removed": rel}


@router.post("/v1/outputs/open")
def open_outputs():
    """Reveal the folder in the OS file manager (local desktop use only)."""
    _us().OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(str(_us().OUTPUT_DIR))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(_us().OUTPUT_DIR)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(_us().OUTPUT_DIR)])
        return {"opened": str(_us().OUTPUT_DIR)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"cannot open: {e}"})


# ---------- model install/uninstall (PC) ----------



class InstallRequest(BaseModel):
    id: str = ""


