# Prism — PC server

FastAPI server with an OpenAI-compatible API and a single-file WebUI.

## Quick start (Windows)

```powershell
cd pc
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy models.json.example models.json      # adjust if needed
python download_models.py --preset odysseus-default
python -m uvicorn unified_server:app --host 127.0.0.1 --port 8101
```

Open `http://127.0.0.1:8101/webui`.

**Easiest:** double-click `Start-Prism.bat` (in the repository root, or in
this `pc/` folder if you downloaded only PC). It finds a Python with the
dependencies, starts the server and opens the WebUI; if port 8101 is occupied
by the old service it uses 8102. Stop with `Stop-Prism.bat`.

Linux/macOS: `./scripts/start-prism.sh` (venv at `pc/.venv` or repo root `.venv`),
stop with `./scripts/stop-prism.sh`.

To run Prism automatically at boot/login on either platform:

```bash
# Linux (systemd user unit, enables lingering so it starts before login)
# macOS (launchd agent)
./pc/scripts/install-prism-service.sh

# Windows, elevated PowerShell
powershell -ExecutionPolicy Bypass -File pc\scripts\install-prism-service.ps1 -RemoveLegacy
```

`-RemoveLegacy` deletes the old `OdysseusDiffusion` and `whisper-api` services.
To remove them later, or on a machine where Prism is already installed:

```powershell
# preview first, then run for real
powershell -ExecutionPolicy Bypass -File pc\scripts\remove-legacy-services.ps1 -WhatIf
powershell -ExecutionPolicy Bypass -File pc\scripts\remove-legacy-services.ps1
```

## GPU (optional)

```powershell
pip install onnxruntime-gpu nvidia-cudnn-cu13       # NVIDIA CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install transformers                            # SmolVLM caption/chat

# AMD: separate env with a non-QDQ model
pip install onnxruntime-directml
```

- `PRISM_ONNX_PROVIDER=auto|cuda|dml|cpu` (default `auto`).
- `PRISM_UPSCALE_ENGINE=auto|torch|onnx|pillow`.
- `/health` reports `device` and `onnx_provider`.
- The INT8 QDQ LCM model cannot use DirectML; AMD stays on CPU unless a
  non-QDQ FP16/FP32 model is exported.

## Models

- `models/` (downloaded by `download_models.py`, never committed).
- `models.json` — local config (copy from `models.json.example`).
- Upscale: `download_models.py --preset upscale`, then
  `python tools/convert_esrgan_onnx.py` for torch-free ONNX inference.
- VLM: `download_models.py --preset vlm`.

## Tests

```powershell
python -m pytest tests -q                  # unit tests (44)
python tools/test_pc_features.py           # e2e against :8200
python tools/test_ffmpeg_presets.py
python tools/test_pc_states.py
python tools/test_model_manager_concurrency.py
```

The e2e drivers expect the server on `http://127.0.0.1:8200` (start uvicorn
with `--port 8200`).

## Layout

| Path | Purpose |
|---|---|
| `unified_server.py` | FastAPI app, endpoints, WebUI route |
| `unified/` | LCM ONNX pipeline, upscale, VLM, FFmpeg, model manager, registry |
| `diffusion_server.py` | legacy Diffusers engine (CUDA) used when torch is installed |
| `webui_unified.html` | WebUI |
| `scripts/` | Start/Stop (`.ps1` + `.sh`), service install, legacy cleanup, firewall, systemd |
| `tools/` | e2e tests, model conversion, reference scripts |
| `tests/` | pytest unit tests |
| `legacy/` | old image-only launcher and scripts (kept for reference) |
| `docs/` | PC operations and contract |

## Service (Windows)

The legacy services (`OdysseusDiffusion`, `whisper-api`) are replaced by the
single `Prism` service. Install it from an elevated PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File pc\scripts\install-prism-service.ps1 -RemoveLegacy
```

Without elevation the same command writes a hidden launcher into the user's
Startup folder instead, which also works but only starts at login.

To delete the old services and their files later:

```powershell
powershell -ExecutionPolicy Bypass -File pc\scripts\remove-legacy-services.ps1 -WhatIf   # preview
powershell -ExecutionPolicy Bypass -File pc\scripts\remove-legacy-services.ps1           # delete
```

## Security and LAN access

By default, the server binds to **127.0.0.1** (localhost only). To allow LAN
access, pass `--host 0.0.0.0` (or use `-Remote` in the PowerShell launchers,
or set `PRISM_HOST=0.0.0.0`).

**WARNING:** The API has **no authentication**. Binding to 0.0.0.0 exposes an
unauthenticated API to the network. Only use LAN binding on a trusted network.
Do **not** expose port 8101 to the internet. Use Tailscale or your LAN firewall
to restrict access.

## Docs

- `docs/ADMIN-SETUP.md` — service install, legacy removal, verification.
- `docs/UNIFIED-OPS.md` — operations, CLI, models, GPU, FFmpeg.
- `docs/UNIFIED-CONTRACT.md` — API contract.
- `../docs/ODYSSEUS-CONTRACT.md` — shared integration contract.
