---
name: lenswright
description: "Operate Lenswright, a local offline OpenAI-compatible AI server on PC (pc/) and Android (android/). Use for starting/stopping/verifying the server, image generation, edits, inpaint, outpaint, colorize, restore, upscale, OCR, caption, VLM chat, transcription, TTS, realtime ASR, live transcription sessions, FFmpeg, batch jobs, model downloads, secrets and client setup. Covers endpoints, model IDs, env vars, honest-503 fallbacks and ADB testing. Trigger on Lenswright, port 8101 or 'is the server running'."
version: 1.0.0
author: DracieVajko
tags: [lenswright, local-ai, server, android, image-generation, ocr, asr, adb]
category: custom
source_repository: https://github.com/DracieVajko/Dracievajko-SkillsSet
source_file: skills/original/lenswright/SKILL.md
license: MIT
local_modifications: original
verification_status: verified
authorship: original
---

# Lenswright — agent operating guide

Lenswright is **two independent products** sharing one API contract:

| Product | Folder | Runtime | UI | When to use it |
|---|---|---|---|---|
| **PC server** | `pc/` | Python + FastAPI + Uvicorn | WebUI at `/webui` | Heavy models (SDXL, FLUX, Z-Image, big OCR/ASR), CUDA GPU, FFmpeg, batch/video jobs |
| **Android app** | `android/` | Kotlin + Ktor + ONNX Runtime + whisper.cpp | native tabs (no WebUI) | Fully offline on the phone, no PC needed; smaller on-device models |

Both expose an **OpenAI-compatible HTTP API on port 8101**, bind `0.0.0.0`, and
use the same response shapes. Everything runs locally and offline *after* the
model is downloaded. Weights are never bundled and are never committed.

Core principles to preserve:
- **Single resident model** — at most one heavy engine loaded at a time; a switch releases the previous one.
- **Honest errors** — unavailable features should return `503 {"error":"will not load", "reason":...}`; beware a few legacy 200 fallbacks (see §6).
- **Lazy models** — download on demand, `.READY` manifest only after preparation; resume + cancel supported.
- **LAN-only, optional token** — no auth by default; set `LENSWRIGHT_API_TOKEN` for bearer auth; never expose to the internet.

**Code is authoritative.** Several docs (`docs/CHECKLIST.md`, `docs/ONDEVICE-PLAN.md`,
`docs/ARCHITECTURE.md`, `pc/README.md`, `pc/docs/UNIFIED-CONTRACT.md`) are stale
relative to the source — e.g. they claim Android remote mode was removed (it is
implemented) and quote a 100 MB upload cap (code uses 500 MB). When a doc and
the code disagree, trust the code and the live `/health`, `/v1/settings`,
`/v1/models/catalog` responses.

---

## 1. Choosing the right path (what / why)

| Goal | Use | Endpoint / note |
|---|---|---|
| Text → image, PC, best quality | PC | `POST /v1/images/generations`; SD models need torch/diffusers, else ONNX LCM |
| Text → image, phone, offline | Android | same endpoint; LCM ONNX QDQ, 256–512 px, ~8–30 s |
| Photo → different scene / style | Both | `POST /v1/images/edits` (img2img, multipart) |
| Fill / replace a region | Both | `POST /v1/images/inpaint` (add/remove/color) |
| Expand canvas | Both | `POST /v1/images/outpaint` |
| Match a pasted element into a background | Both | `POST /v1/images/harmonize` (CPU color match + seam feather) |
| Black & white → color | Both | `POST /v1/images/colorize`; default `deoldify-artistic` |
| Old/damaged photo repair | PC | `POST /v1/images/restore` (Z-Image first, SDXL img2img fallback ≤0.35 strength); `face-restore`, `restore-upscale` |
| Enlarge photo | Both | `POST /v1/images/upscale` (PC: Real-ESRGAN/SwinIR/Z-Image; Android: ncnn) |
| Enlarge video | PC | `POST /v1/videos/upscale` (frame extract + upscale + remux audio) |
| Many images at once | Both | `POST /v1/batch`, then `GET /v1/batch/{id}` |
| Read text from image/document | Both | `POST /v1/images/ocr` (multipart); `POST /v1/documents/extract` for PDF/DOCX/images |
| Describe an image | PC / Android with VLM | `POST /v1/images/caption` |
| Vision Q&A | PC / Android with VLM | `POST /v1/chat/completions` with `image_url` data-URI |
| Speech → text / English | Both | `POST /v1/audio/transcriptions`, `/translations` |
| Realtime microphone ASR | Both | WebSocket `/v1/realtime` (OpenAI transcription session; Cindy client) |
| Live session → two `.txt` files | Both | `POST /v1/live/sessions` (+ `/append`, `/close`, `GET`); separate `mic`/`device` files, **no audio saved** |
| Text → speech | Both | `POST /v1/audio/speech`, `GET /v1/audio/voices` |
| Convert/trim/compress media | PC full, Android partial | PC: eight explicit `POST /v1/ffmpeg/convert-video\|convert-audio\|convert-image\|extract-audio\|trim\|compress\|info\|run` routes; Android: generic `POST /v1/ffmpeg/{op}` (503 stub) |
| Inspect / download artifacts | Both | `GET /v1/outputs`, `GET /v1/outputs/{rel}`, `DELETE ...` |
| Track long jobs | Both | `GET /v1/jobs`, `GET /v1/jobs/{rid}`, `POST /v1/jobs/{rid}/cancel` |
| Install/remove models | Both | `POST /v1/models/install|uninstall|cancel|add|delete`, `GET /v1/models/catalog` |
| Store HF token / LLM provider | Both (Settings UI) | `GET|PUT /v1/settings/secrets`, test endpoints |
| Check / apply update | PC | `GET /v1/update/check`, `POST /v1/update/apply` (restart required) |

Rule of thumb: **PC for CUDA/heavy work, Android for offline phone work**.
Android has a **Local / Remote / Auto** connection mode (`AppConfig.connectionMode`,
`RemoteProxy.kt`): `local` = on-device only; `remote` = forward every request to
the configured PC host; `auto` = **try the remote PC first**, fall back to local
only if forwarding fails (Ktor interceptor in `ApiServer.kt`). The remote host is
set in the UI spinner + host field (e.g. `http://192.168.1.100:8101` or a
Tailscale IP). Remote/auto activate only when the host is non-blank. Keep this
in mind before concluding "Android cannot do X" — in remote/auto it may be
running on the PC.

---

## 2. Server lifecycle

### 2.1 PC — start

One-click (Windows, from repo root):
```powershell
.\Start-Lenswright.bat                 # uses pc\scripts\Start-Lenswright.ps1, opens WebUI
.\Start-Lenswright.bat -Remote         # bind 0.0.0.0 (warns if no LENSWRIGHT_API_TOKEN)
.\Start-Lenswright.bat -Port 8102 -NoBrowser -Cpu
```
Manual (any OS, run from `pc/` so relative model paths resolve):
```powershell
cd pc
python -m uvicorn unified_server:app --host 127.0.0.1 --port 8101
# WebUI: http://127.0.0.1:8101/webui
```
Scripts: `pc/scripts/start-lenswright.sh` (Linux/macOS, `LENSWRIGHT_PORT`, `LENSWRIGHT_HOST`,
`LENSWRIGHT_PYTHON`), `pc/scripts/start-lenswright-autostart.ps1` (logon autostart),
`pc/scripts/install-lenswright-service.ps1` (NSSM Windows service `Lenswright`),
systemd unit under `pc/scripts/systemd/`.

`Start-Lenswright.ps1` discovers a Python that has `fastapi`, `uvicorn`, `onnxruntime`
(prefers `pc/.venv`), falls back to port 8102 if 8101 is busy, and opens the
WebUI unless `-NoBrowser`.

### 2.2 PC — stop

```powershell
.\Stop-Lenswright.bat                  # NOTE: root bat calls pc\scripts\Stop-Unified.ps1
                                  #       which may be missing — prefer the script below
powershell -ExecutionPolicy Bypass -File pc\scripts\Stop-Lenswright.ps1 -Port 8101
```
`Stop-Lenswright.ps1` stops the Windows service, the autostart runner, then whatever
holds the port. Also valid: `Stop-Service Lenswright`, or kill the PID
from `Get-NetTCPConnection -LocalPort 8101 -State Listen`.

### 2.3 PC — verify it is running

```powershell
python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8101/health',timeout=5).read().decode())"
# or: Invoke-RestMethod http://127.0.0.1:8101/health
```
`/health` shows `status`, `device`, `onnx_provider`, `busy`, `queue`,
`in_flight`, `current_model`, `loading`, `audio_model`. `/healthz` is the
lightweight ASR probe used by Cindy. Check the listener directly with
`Get-NetTCPConnection -LocalPort 8101 -State Listen` before concluding "down".

**Gotcha:** with a single Uvicorn worker, one long CPU/GPU inference holds the
GIL; `/health` and even `/healthz` can time out while the process is alive and
the GPU is at 100 %. Always check the process list before assuming a crash.

### 2.4 Android — start / verify

Install and launch:
```powershell
cd android
.\gradlew.bat :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n eu.odysseus.diffusion/.MainActivity
# Server tab -> START, or headless:
adb shell am start -n eu.odysseus.diffusion/.MainActivity --ez start_server true
adb forward tcp:8101 tcp:8101        # map phone:8101 to PC:8101 for testing
```
Then verify exactly like the PC (`GET http://127.0.0.1:8101/health` through the
forward). The app runs a **foreground Ktor service**, so it survives
backgrounding. The UI reports `engines: torch=false cuda=false onnx=true
whisper=true` in Settings.

Stop Android server: tap **STOP** in the Server tab, or
`adb shell am force-stop eu.odysseus.diffusion`.

To watch logs: `adb logcat` (filter `LcmOnnxEngine`, `RealEsrganNcnn`,
`realesrgan_jni`). If a foreground-service start is blocked on Android 12+
(`ForegroundServiceStartNotAllowedException`), the server still runs without the
notification; opening the app and pressing START fixes the UI state.

---

## 3. Endpoints and request shapes

Base: `http://<host>:8101`. Images/audio return base64 JSON; most long jobs add
`request_id` and an `outputs` list of saved files. All paths work on both
products unless marked **PC only** or **Android only**.

### 3.1 Images

| Endpoint | Request | Key fields / notes |
|---|---|---|
| `POST /v1/images/generations` | JSON | `prompt`, `negative_prompt`, `guidance` (1.0 = LCM, >1 = CFG), `size` (`WxH`), `quality`, `steps`, `n` (1–4, distinct seeds), `model`, `request_id`. Square canvas = larger requested side clamped 256–512; response reports `size` + `requested_size`. |
| `POST /v1/images/edits` | multipart | `image` + `prompt`, `model`, `n`, `size`, `quality`, `strength`, `request_id` |
| `POST /v1/images/inpaint` | JSON | `image`, `mask`, `prompt`, `strength`, `mode=add\|remove\|color`, `color`, `steps`, `model`. `color` is non-AI HSV recolor preserving luminance. Black mask = untouched, white = repainted. |
| `POST /v1/images/outpaint` | JSON | `image`, `prompt`, `margin` or `left/right/top/bottom`, `strength`, `steps`. Edge-replicate padding + border mask. Caps 2048 px/side, 4 MP. |
| `POST /v1/images/harmonize` | JSON | `image` (+ optional `mask`), `prompt`, `strength`. CPU color match + seam feather; no mask = mild auto-level. `engine=cpu-harmonize`. |
| `POST /v1/images/colorize` | JSON | `image`, `model`: `deoldify-artistic` (default, most chroma), `ddcolor-full`, `ddcolor-artistic`. Preserves resolution + luminance. |
| `POST /v1/images/restore` | JSON | `image`, `prompt`, `strength`, `steps`, `guidance`, `image_guidance`. Z-Image Turbo first, SDXL img2img fallback (strength ≤ 0.35). 503 if neither installed. |
| `POST /v1/images/face-restore` | JSON | face-focused restore; 503 when no engine |
| `POST /v1/images/restore-upscale` | JSON | combined restore + upscale |
| `POST /v1/images/upscale` | JSON | `image`, `scale` 2/3/4, `model`: `RealESRGAN_x4v3` (default), `swinir-realsr-x4`, `z-image-turbo`; optional `prompt`, `strength` for Z-Image. Unknown model = 400. |
| `POST /v1/videos/upscale` | multipart | `file`, `scale`, `model`, `fps`, `prompt`, `strength`. Deterministic engines ≤900 frames; Z-Image ≤24 frames (400 above). Remuxes audio. |
| `POST /v1/batch` | JSON | `{"jobs":[{"type":"upscale","image":"<b64>","scale":2,"model":...,"prompt":...,"strength":...}]}`; poll `GET /v1/batch/{id}`, cancel `POST /v1/batch/{id}/cancel`. |

Quick example:
```bash
curl -s http://127.0.0.1:8101/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red cube","size":"512x512","model":"lcm-dreamshaper-int8-onnx"}' \
  | python -c "import sys,json,base64;d=json.load(sys.stdin);open('out.png','wb').write(base64.b64decode(d['data'][0]['b64_json']))"
```

### 3.2 Vision, OCR, documents

| Endpoint | Request | Notes |
|---|---|---|
| `POST /v1/images/ocr` | multipart `file` + `model`, `language`, `task=ocr\|describe` | PC models: `nemotron-ocr-0`, `granite-docling-258m`, `qwen3-vl-2b`, `ovis-ocr2`, `glm-ocr`, `paddleocr-vl-1.6`, `lightonocr-2-1b`, `qianfan-ocr`. Document-only models reject `describe` (400). Android: `xiaomi-ocr-0`, honest 503 for others. |
| `POST /v1/images/caption` | JSON `image`, `prompt`, `max_tokens`, `request_id` | VLM; 503 without weights/runtime |
| `POST /v1/chat/completions` | OpenAI shape; messages content is text or `[{type:text\|image_url,image_url:{url:"data:..."}}]` | PC: Qwen3-VL-2B. Android: SmolVLM2. `model` may be honored or mapped to the resident VLM; 503 without runtime. |
| `POST /v1/documents/extract` | multipart / JSON | txt/md/csv, DOCX (zip XML), PDF (text layer or render+OCR), images (OCR) |

OCR gotcha: for Qwen-VL the **chat template must insert image tokens**. A bare
prompt yields zero image tokens and an empty string. Keep
`apply_chat_template(tokenize=True, return_dict=True)` with the image inside the
message content (transformers 5.x).

### 3.3 Audio and realtime

| Endpoint | Request | Notes |
|---|---|---|
| `POST /v1/audio/transcriptions` | multipart `file` (or `url`) + `model`, `language`, `device` | PC: faster-whisper local (tiny…large/turbo) or proxy to `UNIFIED_WHISPER_URL`; Android: whisper.cpp JNI, 30 s chunking. Long PC jobs use 5-min chunks with `initial_prompt`. |
| `POST /v1/audio/translations` | same | → English |
| `POST /v1/audio/transcribe-batch` | JSON list | batch ASR |
| `POST /v1/audio/speech` | JSON `text`, `voice`, `language`, `format` | PC: Supertonic 3 (31 languages incl. Slovak) + KittenTTS (English, 8 voices) via sherpa-onnx; auto-downloads. Android: system TextToSpeech → WAV, 503 if language unavailable. |
| `GET /v1/audio/voices` | — | available voices |
| WebSocket `/v1/realtime` | OpenAI transcription session | client: `session.update`, `input_audio_buffer.append` (base64 PCM16), `commit`, `clear`, `session.finish`; server: `session.updated`, `input_audio_buffer.committed`, `conversation.item.input_audio_transcription.completed`. 24 kHz → 16 kHz resample. No partial deltas in v1. |

```bash
curl -s -F "file=@jfk.wav" -F "language=en" http://127.0.0.1:8101/v1/audio/transcriptions
```

**Live sessions (file-only).** `POST /v1/live/sessions` accepts optional
`language`, `model`, `label` and returns `id`, `sources:["mic","device"]` plus
per-source `descriptors` of shape
`{name, kind:"text", size, path:"text/<name>", url:"/v1/outputs/text/<name>"}`.
While capturing, send each final transcript to
`POST /v1/live/sessions/{id}/append` with `{source, text, item_id?, elapsed_ms?}`;
`POST /v1/live/sessions/{id}/close` flushes both files, and
`GET /v1/live/sessions/{id}` reports state and line counts. Two **separate** files
are written under `pc/output/text/` (PC) or `files/output/text/` (Android) and
appear in `GET /v1/outputs`. **No audio is recorded or saved** — PCM stays in
client memory, only text crosses the unchanged `/v1/realtime` socket (Cindy needs
no change), and neither client nor server creates a `MediaRecorder` or audio file.
Appends are flushed immediately and deduplicated on `item_id`.

Limits: PC Live capture is **localhost-only** (`localhost`/`127.0.0.1`; secure
context, no TLS) and captures one shared **browser tab**, not system audio — on
LAN/Tailscale the controls are disabled. Android device audio uses
`MediaProjection` (apps that disallow playback capture produce silence); all
`/v1/live/*` calls and live sockets bypass Local/Remote/Auto forwarding and run
on the phone.

### 3.4 Media (FFmpeg)

`GET /v1/ffmpeg/presets` → groups: video (H.264/H.265 HQ/fast, mp4-small,
mp4-discord <10 MB, webm, gif), audio (mp3 128/192/320, aac, wav-16k, opus,
flac), image (jpg-95/85, png, webp, avif). Operations:
`POST /v1/ffmpeg/{convert-video|convert-audio|convert-image|extract-audio|trim|compress|info|run}`
(multipart). PC uses real ffmpeg; Android uses MediaCodec/MediaMuxer plus
libmp3lame — FLAC/Opus, AVIF, VP9/WebM and video re-encode return honest 503.

### 3.5 Models, jobs, outputs, settings

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | model list with `kind`, `installed`, `size_mb`, `install_state`; also `currently_loaded_model`, `loading`, `missing_models`, defaults |
| `GET /v1/models/catalog` | full catalog with license, install progress/error; grouped by kind |
| `POST /v1/models/install` | `{"id":...}`; states `idle → downloading → preparing → done\|error`; preparation hook runs before `.READY` |
| `GET /v1/models/progress/{id}`, `/status/{id}`, `/verify/{id}` | progress / state / file verification |
| `POST /v1/models/cancel`, `/uninstall`, `/add`, `/delete` | management (path-guarded to `models/`) |
| `GET /v1/models/find`, `/v1/models/custom` | HF search / custom models |
| `GET /v1/jobs`, `/v1/jobs/{rid}`, `POST /v1/jobs/{rid}/cancel` | long jobs (409 `{"error":"cancelled"}`) |
| `GET /v1/outputs`, `/v1/outputs/{rel}`, `DELETE /v1/outputs/{rel}`, `POST /v1/outputs/open` | saved artifacts; path traversal refused |
| `GET /v1/settings` | version, port, models_dir, defaults, `engines:{torch,cuda,ffmpeg,ffprobe,ytdlp}` |
| `GET|PUT /v1/settings/secrets`, `POST /v1/settings/secrets/test/{hf,llm}` | masked secrets + validation |
| `GET /v1/update/check`, `POST /v1/update/apply` | PC auto-update (git pull; restart required) |
| `GET /v1/images/progress/{request_id}` | legacy generation progress |

---

## 4. Model selection quick reference

PC image models (`pc/models.json`, default `DreamShaper_8`):
`DreamShaper_8`, `AbsoluteReality_1.8`, `AnimagineXL_3.1`, `JuggernautXL_v9`,
`SDXL_Turbo`, `NoobAI-XL-Vpred`, `amazing-logos-v3`. SD/SDXL need torch +
diffusers; otherwise the ONNX LCM model runs (`lcm-dreamshaper-int8-onnx`).

Installable catalog IDs (see `GET /v1/models/catalog` for sizes/licenses):
- **Upscale/restore:** `RealESRGAN_x4v3`, `swinir-realsr-x4`, `z-image-turbo`
  (fp8 variant `z-image-turbo-fp8` is the practical size), `ddcolor-full`,
  `ddcolor-artistic`, `deoldify-artistic`, `flux-kontext-dev` (gated, needs HF
  token; may not run on low-RAM machines), `flux-kontext-dev-fp8` (ungated).
- **OCR:** the 8 IDs in §3.2.
- **ASR:** whisper sizes + `cohere-transcribe`, `granite-speech-470m|2b`,
  `parakeet-v3|v2|unified-en`, `nb-whisper-medium`, `ivrit-whisper-turbo`.
- **TTS:** `supertonic-3-tts-int8` (31 languages incl. SK),
  `KittenML/kitten-tts-mini-0.8` (English, 8 voices, auto-download).

Android installed set (typical):
`whisper-large-v3-turbo-q5_0`, `lcm-dreamshaper-int8-onnx`,
`realesrgan-x4plus-ncnn`, `smolvlm2-500m-gguf` (or `gemma4-e2b-gguf`),
`xiaomi-ocr-0`, `deoldify-artistic`, `ddcolor-full`, `ddcolor-artistic`.

When a model is missing, install it first; do not claim success. Prefer the
smaller/faster model that satisfies the task (e.g. `nemotron-ocr-0` for quick
text, `glm-ocr` for layout/reading order, `qwen3-vl-2b` for description).

---

## 5. Configuration (environment variables)

Set before starting the server (PC) or via the Android Settings UI.

| Variable | Effect |
|---|---|
| `LENSWRIGHT_API_TOKEN` | Require `Authorization: Bearer <token>` on `/v1/*`; set it before `-Remote` / `0.0.0.0` |
| `LENSWRIGHT_ONNX_PROVIDER` | `cpu` forces CPU even with CUDA (fallback for 6 GB cards); default auto (CUDA → CPU) |
| `LENSWRIGHT_ASR_MODEL` | default ASR model (e.g. `turbo`) |
| `LENSWRIGHT_ASR_ENGINE` | force ASR engine |
| `WHISPER_DEVICE` | `cuda` opts into GPU whisper (default CPU int8) |
| `UNIFIED_WHISPER_URL` | legacy WhisperApi proxy URL (default `http://127.0.0.1:4444`) |
| `UNIFIED_IDLE_UNLOAD_SECONDS` | PC idle unload delay (default 900 s) |
| `UNIFIED_NO_ENGINE` | `1` skips engine init; any other truthy value disables VLM/caption/chat → 503 |
| `LENSWRIGHT_OCR_DEVICE` | `auto\|cpu\|cuda` for OCR/VLM; `cuda` avoids cross-device hooks |
| `LENSWRIGHT_NEMOTRON_DEVICE` | device override for Nemotron OCR |
| `LENSWRIGHT_RESTORE_ENGINE` | `sd` forces SDXL restore; default Z-Image first |
| `LENSWRIGHT_RESTORE_MODEL` | override restore model |
| `LENSWRIGHT_UPSCALE_ENGINE` | `torch\|onnx\|pillow` |
| `LENSWRIGHT_SWINIR_MAX_SIDE`, `LENSWRIGHT_ZIMAGE_TILE`, `LENSWRIGHT_ZIMAGE_OVERLAP`, `LENSWRIGHT_ZIMAGE_MAX_TILES` | Z-Image / SwinIR tiling caps |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | gated model downloads (or store in secrets) |
| `LENSWRIGHT_LLM_PROVIDER`, `LENSWRIGHT_LLM_BASE_URL`, `LENSWRIGHT_LLM_MODEL`, `LENSWRIGHT_LLM_API_KEY`, `LENSWRIGHT_LLM_ENDPOINT` | LLM provider (also stored in secrets) |
| `LENSWRIGHT_LLM_MAX_CONCURRENT`, `LENSWRIGHT_LLM_POLL_INTERVAL`, `LENSWRIGHT_LLM_WAIT_TIMEOUT` | LLM queue behavior (defaults 1 / 120 / 3600) |
| `LENSWRIGHT_PROMPT_<OPERATION>` | override LLM prompt for `summary\|cleanup\|grammar\|list\|translate\|rewrite\|meeting_notes` |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OLLAMA_BASE_URL`, `LM_STUDIO_BASE_URL` | external LLM providers used by action extraction (Lenswright does not host LLMs) |

Secrets are stored in **`pc/settings.local.json`** (git-ignored) and exposed via
`/v1/settings/secrets` masked as `****abcd`; masked values sent back are ignored
so a refresh cannot wipe a token. Android stores them in app preferences
(Settings → API & model settings). Token is only sent to HF or the configured
LLM base URL.

NSFW safety checker is **ON by default** (`EngineConfig.disable_safety_checker =
False`). Disable only deliberately with the `--disable-safety-checker` CLI flag
(LAN-only use — it false-positives to black images); `/v1/images/restore` sets it
off for the SD fallback. Blacked images are logged, not reported as success.

---

## 6. Honest 503, legacy 200 fallbacks, and known limitations

Most unavailable features return a clear 503. Genuine 503 cases:
- `face-restore` (GFPGAN removed) → **always 503** with a hint to use `restore`/`colorize`;
- missing model / weights → 503 with `hint` (`colorize`, `upscale`, `restore`,
  `restore-upscale`, `outpaint`, `ocr`, `caption`, `chat`, TTS); download via
  `/v1/models/install`;
- `UNIFIED_NO_ENGINE` set → caption/chat 503; text-only chat without LLM provider → 503;
- video upscale without ffmpeg/ffprobe → 503; 400 on frame cap (>900, Z-Image >24);
- RAM/VRAM insufficient → Android `requireMemory()` refuses to *load* an engine
  with numbers (503) rather than letting LMKD kill the process;
- resident gate busy → 503 `{"error":"loading model","retry_after":30}`.

**Legacy 200 fallbacks — inspect the response, not just the status code:**
- `generations` / `edits`: when neither diffusers nor LCM works, PC returns a
  64×64 placeholder PNG with `engine:"placeholder"` — not a real generation;
- `inpaint` / `harmonize`: on engine failure they can echo the input image with
  `engine:"echo"`;
- `transcriptions`: if local whisper fails and the `:4444` proxy is down, the
  text may be `"[unified stub — ...]"` with `engine:"stub"`;
- `translations` does **not** run a translation engine — it delegates to
  `transcriptions()` with `language=None` and returns only `{"text":...}`.

Never report such a 200 as a successful job. Check `engine`, image dimensions,
and text content against the request.

Operational gotchas learned the hard way:
- **One worker** — a single long job blocks the event loop; use the process list
  if health times out.
- **Model switching** can crash a process on some GPU/driver combinations; hold
  one model when testing, or set `LENSWRIGHT_OCR_DEVICE=cuda` and keep the GPU free.
- **Driver resets** (`nvlddmkm` Event 153 on Windows) can surface as native
  `0xc0000005` in `torch_cpu.dll` — that is a GPU driver issue, not a Python bug.
- **Android image sizes** must be multiples of 64 (256–512); 288 fails in the
  UNet concat. `lcmSize()` rounds for you.
- **Android upscale** long side capped at 1536 px (400 px otherwise); ncnn runs
  2 threads at background priority to keep the phone responsive.
- **Android engines** are released after ~45 s idle, with a 15 s watchdog below
  1.5 GB available RAM; PC keeps them ~10 min.

---

## 7. Attaching clients

```python
from openai import OpenAI
client = OpenAI(base_url="http://<host>:8101/v1", api_key="not-needed")
client.images.generate(model="lcm-dreamshaper-int8-onnx", prompt="a red cube", size="512x512")
```

- **Cindy** → realtime WebSocket `/v1/realtime` ("Custom realtime ASR → OpenAI transcription"). Live sessions are a separate HTTP layer beside it; the socket protocol is unchanged.
- **Odysseus** → `endpoint:<id>` HTTP STT against `/v1/audio/transcriptions`.
- **OpenAI SDK / any compatible client** → `/v1/*`.
- Claude Code / Codex CLI built-in voice cannot use a custom ASR endpoint.

---

## 8. Verification checklist (before claiming success)

1. Start server; `GET /health` returns 200 with `status:"ok"`.
2. `GET /v1/models` / `/v1/models/catalog` lists the model you need as installed.
3. Run one real request with a known input (e.g. `testdata/audio/jfk.wav`,
   `testdata/images/dense_market_street.jpg`) and inspect the **output**, not just
   the status code: non-empty text, valid PNG dimensions, expected engine name.
4. Confirm `outputs` file exists under `pc/output/` (PC) or app external `files/output/`
   (Android) when the client needs artifacts.
5. For Android via ADB: `adb forward tcp:8101 tcp:8101`, then the same HTTP checks.
6. Run the suites when changing code: `cd pc && python -m pytest -q` (PC);
   `cd android && .\gradlew.bat test connectedAndroidTest` (unit + device).

Keep changes consistent with the two-product architecture, English-only text,
and honest errors. Update `docs/CHECKLIST.md` after meaningful changes.
