# Prism

On-device AI server for image generation, transcription, OCR and upscaling — with a PC WebUI and a fully native Android app. Both run locally and offline, expose the same OpenAI-compatible API on port 8101, and never bundle model weights.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version: 0.7.0](https://img.shields.io/badge/version-0.7.0-blue.svg)](CHANGELOG.md)
[![Sponsor on GitHub](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=github)](https://github.com/sponsors/DracieVajko)
[![PC CI](https://img.shields.io/github/actions/workflow/status/DracieVajko/Prism/ci.yml?label=PC%20CI)](.github/workflows/ci.yml)

## What Prism is

Prism is **two independent products in one repository**, sharing a single API contract. You only need the folder for the product you want to run — they do not depend on each other.

| Product | Folder | Runtime | UI | Dependencies |
|---|---|---|---|---|
| **PC server + WebUI** | `pc/` | Python 3 + FastAPI + Uvicorn | Single-file WebUI (`/webui`) | torch/diffusers (optional), onnxruntime, faster-whisper, ffmpeg |
| **Android app** | `android/` | Kotlin + Ktor + ONNX Runtime + whisper.cpp | Native tabs (no WebUI) | none at runtime; models downloaded lazily |

Both expose an **OpenAI-compatible HTTP API on port 8101**, so external clients (OpenAI SDK, custom scripts, other AI tools) can attach over LAN:

```python
from openai import OpenAI
client = OpenAI(base_url="http://<host>:8101/v1", api_key="not-needed")
client.images.generate(model="lcm-dreamshaper-int8-onnx", prompt="a red cube", size="512x512")
```

```bash
curl -F "file=@audio.wav" -F "language=en" http://<host>:8101/v1/audio/transcriptions
```

## Problem being solved

Running local AI should not require cloud accounts, subscriptions, or sending private data to third parties. Prism lets you run image generation, transcription, OCR and upscaling entirely on your own hardware — PC or phone — with models downloaded on demand and removed when idle. Features that cannot run on the current machine answer `503` with an honest reason instead of faking output.

## Main features

### PC server (`pc/`)

- **Image:** text-to-image, image-to-image (edits), inpaint with mask, harmonize, outpaint (edge-replicated padding), upscale (Real-ESRGAN / Pillow fallback)
- **Vision:** caption and chat via SmolVLM-500M
- **Audio:** transcription and translation via faster-whisper (tiny/base/small/medium/turbo/large) or NVIDIA Nemotron 3.5 ASR 0.6B
- **OCR:** Xiaomi-OCR-0 via LiteRT-LM
- **Media:** FFmpeg presets (convert video/audio/image, extract audio, trim, compress, info)
- **Models:** lazy download from Hugging Face / direct URLs, resume, cancellation, `.READY` manifests
- **WebUI:** single-file HTML/CSS/JS at `/webui`, dark/light theme, model catalog, outputs gallery, job queue
- **Service:** auto-start at boot/login via systemd (Linux), launchd (macOS), NSSM (Windows)

### Android app (`android/`)

- Fully native, **no WebUI, no PC dependency** — everything runs on the phone
- **Image:** text-to-image, inpaint, harmonize via ONNX Runtime LCM (QDQ, NCNN/Vulkan)
- **Audio:** transcription and translation via whisper.cpp JNI (30 s chunks, auto-split)
- **OCR:** Xiaomi-OCR-0 via LiteRT-LM
- **Caption/chat:** 503 until the VLM runtime is bundled (documented limitation)
- **Models:** downloaded lazily into app storage, never in the APK
- **Server:** foreground Ktor service, keeps running when backgrounded
- **Modes:** Local only (Local / Remote / Auto were removed by design)

### Shared principles

- **Local and offline:** after model download, no network is needed
- **Single resident model:** at most one heavy engine loaded at a time (VRAM/RAM budget)
- **Honest errors:** unavailable features return `503 will-not-load` with a reason, never faked
- **Lazy models:** weights are never bundled; downloaded on demand
- **LAN-only, no authentication:** do not expose ports to the internet; use Tailscale or your LAN firewall

## Supported platforms

| Platform | PC server | Android app |
|---|---|---|
| Windows | ✅ (Python 3.11+, NSSM service) | ✅ (APK, arm64-v8a, Android 10+) |
| Linux | ✅ (systemd user unit) | ❌ |
| macOS | ✅ (launchd agent) | ❌ |
| GPU | CUDA / DirectML via ONNX Runtime; torch CUDA if installed | NNAPI (GPU/NPU) via ONNX Runtime |
| CPU | ✅ (ONNX CPU, fp32/int8) | ✅ (XNNPACK CPU) |

## Quick start

### PC

```powershell
# Windows
cd pc
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_models.py --preset odysseus-default   # optional, downloads models
python -m uvicorn unified_server:app --host 0.0.0.0 --port 8101
```

Open `http://127.0.0.1:8101/webui`.

**Easiest on Windows:** double-click `Start-Prism.bat` (repository root). It finds a Python with dependencies, starts the server and opens the WebUI; if port 8101 is occupied it falls back to 8102. Stop with `Stop-Prism.bat`.

Linux/macOS:

```bash
cd pc
./scripts/start-prism.sh          # venv at pc/.venv or repo root .venv
./scripts/stop-prism.sh
```

### Android

Install the APK from GitHub Releases (push a tag → CI builds and attaches `prism-vX.Y.Z-release.apk`), or build locally:

```powershell
cd android
.\gradlew.bat :app:assembleDebug
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

Open the app, go to the **Server** tab, press **START**, then download the models you need from the model list. The app is fully offline afterwards.

## GPU (optional, PC)

```powershell
pip install onnxruntime-gpu nvidia-cudnn-cu13       # NVIDIA CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install transformers                              # SmolVLM caption/chat

# AMD: separate environment with a non-QDQ model
pip install onnxruntime-directml
```

Relevant env vars:
- `PRISM_ONNX_PROVIDER=auto|cuda|dml|cpu` (default `auto`)
- `PRISM_UPSCALE_ENGINE=auto|torch|onnx|pillow`
- `WHISPER_DEVICE=cuda` (default `cpu`; CUDA 12 DLLs required)
- `PRISM_LLM_MODEL=gemma4:12b` (external LLM provider for post-processing)
- `PRISM_LLM_MAX_CONCURRENT=1` (local model slot gate)

## Configuration

### PC
- `pc/models.json` — local model configuration (copy from `pc/models.json.example`)
- `pc/models/` — downloaded weights (never committed)
- Env vars are read at runtime; see `pc/docs/UNIFIED-OPS.md` for the full list

### Android
- Settings in the app: theme (System/Light/Dark), language (en/sk/cs/de), acceleration (Auto/CPU/GPU)
- Models are managed in the in-app model catalog; downloads use HTTP Range resume with `.READY` manifests

## Usage examples

Generate an image:

```bash
curl -X POST http://127.0.0.1:8101/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cozy mountain cabin at sunset","n":1,"size":"512x512","quality":"low","request_id":"t1"}'
```

Transcribe audio:

```bash
curl -X POST http://127.0.0.1:8101/v1/audio/transcriptions \
  -F "file=@recording.wav" -F "language=en"
```

List models:

```bash
curl http://127.0.0.1:8101/v1/models
```

## Development setup

```bash
# PC tests
cd pc
python -m pytest tests -q

# Android build
cd android
.\gradlew.bat :app:assembleDebug   # requires JDK 17, Gradle 8.10.2
```

## Roadmap

- Android: edits (img2img), inpaint, harmonize, upscale (CPU fallback first, then ONNX), FFmpeg tools, caption/chat via SmolVLM ONNX
- PC: QNN / Hexagon NPU EP for the LCM UNet on Android, Silero VAD before whisper.cpp, text-embedding LRU cache, LAN auth token
- Research-only: MobileDiffusion / SnapGen (sub-second mobile), Sherpa-ONNX as Whisper backend, Moondream2 as caption alternative

See `docs/ONDEVICE-PLAN.md` and `docs/RESEARCH-IMPROVEMENTS.md` for the full plan.

## Known limitations

- Android VLM (caption/chat): runtime not bundled → `503`
- Android upscale: Real-ESRGAN engine not implemented; endpoint returns a stub or `503`
- Android FFmpeg: no binary → `503`
- PC without torch: SDXL / AbsoluteReality / Animagine / Juggernaut cannot run; only LCM ONNX is available
- PC CPU-only onnxruntime: image generation is CPU-bound; install `onnxruntime-gpu` or `onnxruntime-directml` for GPU
- MP3 192/320 on a 16 kHz mono source are capped by libmp3lame (~160 kbps)
- Two 30B-A4B Nemotron variants are PC download-only (NVFP4 needs Blackwell GPU, FP8 needs ~40 GB VRAM or 64 GB+ RAM)

## Contribution guidelines

- All code, docs, commit messages, UI strings and log messages must be **English only** (see `AGENTS.md`).
- Android UI translations live in `res/values-sk`, `values-cs`, `values-de` (base `values/strings.xml` stays English).
- If a feature cannot run on Android yet, mark it **experimental**, keep the endpoint honest (`503` with a clear reason), and document the plan in `docs/ONDEVICE-PLAN.md`. Never fake success.
- Models are downloaded lazily; never bundle weights in the APK or repo.
- Update `docs/CHECKLIST.md` after every meaningful change.
- Create a branch, do not push directly to `main` without review.

## Security

LAN-only, no authentication. Do not expose ports 8101/8102 to the internet; bind Tailscale or your LAN. The NSFW safety checker is disabled in the launchers.

## License

MIT — see `LICENSE`. Copyright (c) 2026 DracieVajko.

---

## Support this work

Prism is developed and maintained by one person. Sponsoring helps keep the project local-first, offline-capable and free of cloud dependencies.

Your support goes toward:

- **Feature development** — new on-device capabilities (Edits, Inpaint, Caption, FFmpeg tools on Android; QNN NPU, Silero VAD on PC)
- **Platform compatibility** — keeping the PC server and Android app working across Windows, Linux, macOS and arm64-v8a phones
- **Maintenance** — dependency updates, honest-error contract, single-resident model discipline
- **Bug fixes** — triaging issues, fixing crashes and 503 edge cases
- **Testing** — PC unit suite (39 tests), Android live suite (27 checks), CI on every push
- **Documentation** — architecture, operations, contract and on-device plan docs

[![Sponsor on GitHub](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=github)](https://github.com/sponsors/DracieVajko)

### Sponsorship tiers

| Tier | Suggested | What it supports |
|---|---|---|
| **User** | $5/month | General maintenance and dependency updates |
| **Contributor** | $15/month | Bug fixes and CI reliability (PC + Android test suites) |
| **Sponsor** | $50/month | New feature development (roadmap items above) |
| **Partner** | $150/month | Priority platform compatibility work and direct contact for integration questions |

Tiers are **suggested amounts only** — sponsorship does not grant exclusive features, guaranteed implementation dates, or priority support. All development remains public in this repository.
