# Changelog

## 0.8.0 (2026-11-26) — renamed to Lenswright

- **Renamed** the project to Lenswright. The previous name was shared with dozens
  of unrelated tools — 76 000+ GitHub repositories matched it — so the project
  could not be found. The API contract is untouched: same port 8101, same
  `/v1/*` paths, same model IDs, same 503 shapes.
- **Why Lenswright:** *lens* is the scanning side (OCR, caption, upscale,
  restore), *wright* is the maker — a lenswright grinds lenses.
- **Environment variables:** the old `PRISM_*` prefix became `LENSWRIGHT_*`
  (48 names, including the seven `PROMPT_*` overrides and the four keystore
  variables).
- **Backwards compatibility:** `pc/unified/__init__.py` copies every
  old-prefixed variable to its new name at import time when the new name is
  unset, so existing `settings.local.json`, `.bat`/`.sh` launchers and systemd
  units keep working. That module is the only place in the code that still
  spells the old prefix, and `release.yml` keeps the old secret names as a
  fallback; both go away in 0.9.0.
- **Files and services renamed:** `Start-Lenswright.bat`, `Stop-Lenswright.bat`,
  `pc/scripts/start-lenswright.sh`, `install-lenswright-service.{ps1,sh}`,
  `lenswright.service`, `lenswright.env.example`, and the Windows service
  (`OdysseusDiffusion` -> `Lenswright`). Re-run the installer to re-register it.
- **Android:** launcher name `Lenswright` (`app_name`, `android:label`) and the
  nine colour tokens `lenswright_*`. `applicationId` deliberately stays
  `eu.odysseus.diffusion` so existing installs, settings and downloaded models
  survive; it changes at 1.0.
- **Skill:** `skills/original/lenswright/SKILL.md` with new trigger words.
- **GitHub:** repository is now `DracieVajko/lenswright`.

## 0.7.0 (2026-10-27) — Lenswright

- **Repository split:** `pc/` is the PC product (server, `unified/`, WebUI, models,
  tests, scripts, docs), `android/` is the on-device app. A user needs only one
  of them; root keeps README/AGENTS/LICENSE plus shared `docs/` and `testdata/`.
- **PC WebUI ("Lenswright"):**
  - fixed the CSS token cycle (`--bg: var(--bg)`) that made every input
    white-on-white; real dark/light palettes
  - `/v1/models` reports `kind` + `installed` from the installable catalog, so
    the model dropdowns are per kind (image / whisper / upscale) and no longer
    duplicated; the image-model select only shows on the image tabs
  - Download buttons for every model that is not installed, including whisper
    sizes (verified manifest, exact byte sizes, resume across HF mirrors)
- **PC engine fixes (found by the new content-level suite):** the diffusion
  registry was empty when embedded, every generation silently fell back to the
  ONNX LCM path; `models.json` still pointed at the pre-split root; `diffusers`
  and `accelerate` were missing. Responses now report the real output size and
  the `engine` that produced it.
- **New endpoint:** `POST /v1/images/outpaint` (PC and Android) — edge-replicated
  padding, border mask, 2048 px/side and 4 MP caps.
- **Transcription:** local faster-whisper models (`small`, `turbo`, `large`) with
  one-click download, CPU int8 by default (`WHISPER_DEVICE=cuda` opts in), one
  pass over long files instead of 30 s re-splitting, and `audio_loaded` in
  `/health` reflecting what is really resident.
- **Android:** NCNN Real-ESRGAN upscale with deterministic 4x weight selection,
  outpaint endpoint, on-device vision chat via SmolVLM, honest 503 for the VLM
  runtime and ffmpeg when they are not bundled, ORT serialisation + resident
  LCM engine (no more SIGSEGV under concurrent requests).
- **Tooling:** `Start-Lenswright.bat` / `Stop-Lenswright.bat` launchers, `pc/tools/test_pc_all.py`
  (39 content-level checks), `--only` filter for fast subsets.
- **Hygiene:** `.gitignore` covers secrets, caches, model formats and archives;
  `nssm.exe` provenance documented (`pc/scripts/NSSM.md`, pinned SHA256); stale
  `*_tmp.ps1` tools renamed; unused CLI/quantisation pins dropped; contract docs
  cross-referenced (`ODYSSEUS-CONTRACT` marked superseded, `UNIFIED-CONTRACT`
  gains outpaint).

## 0.6.0 (unreleased)
- No size cap on edits/inpaint/harmonize (DoS guard 8192px); big canvases
  force CPU offload automatically; generations keep the 2048px VRAM-fit cap
- Load planning counts canvas activations; inference OOM → 503 `will not load`

## 0.5.0 (unreleased)
- VRAM-aware loading: fits VRAM → CUDA; VRAM short but RAM fits → CPU fp32
  fallback (slow but works); fits nowhere → immediate 503 `will not load`
  (no more hanging into client timeouts). OOM during load → same 503.
- `device` (cuda/cpu) in `/health`; derived pipes follow the active dtype

## 0.4.0 (unreleased)
- Everything in English: docs, WebUI, scripts, contract
- LAN setup alongside Tailscale (`-TailscaleIp` = any IP, LAN firewall example)
- Detailed Odysseus wiring guide (endpoint + re-probe + timeouts)
- Model swap: RealisticVision out, AbsoluteReality_1.8 in (all references)
- Portable launchers (relative paths, Tailscale auto-detect, env overrides)

## 0.3.0 (unreleased)
- Public-repo release: `models.json.example`, `systemd/` unit + env example,
  `LICENSE` (MIT), Linux quickstart (incl. honest RPi5/ARM64 note)
- Secrets sweep: no IPs/paths in launchers (Tailscale auto-detect, `$ServerDir`
  relative paths, `DIFFUSION_PORT/HOST/MODELS_JSON` env overrides)
- `install_service_admin.ps1`: `-TailscaleIp` param, portable paths

## 0.2.0 (unreleased)
- Release package: README, INSTALL, OPERATIONS, TROUBLESHOOTING,
  `docs/ODYSSEUS-CONTRACT.md`, `requirements.txt` (pinned), `.gitignore`
- CLI: `generate_image.py` subcommands (`txt2img|img2img|inpaint|models|doctor|health`)
- `download_models.py`: presets (`odysseus-default|sd15|sdxl`), `--list`,
  Civitai `--civitai` download with token, size verification
- Model auto-discovery (`auto_discover`, `guessed` flag) + `performance`
  presets (`auto|low|mid|high|cpu`, VRAM+RAM detection, low-VRAM offload rule)
- `--log-file` / `--log-level`, `request_id` in log lines, WebUI: strength
  slider, busy/queue status, settings in localStorage, refresh button
- `high` quality mapping fixed to 30 steps (was 20, less than medium)
- install_service_admin: NSSM env fixed (per-VAR entries, was one `;` blob)

## 0.1.0 (unreleased)
- Lazy-loading OpenAI-compatible image server (port 8101): generations, edits
  (native img2img), inpaint, harmonize, progress polling, `/webui`
- Model registry (`models.json`) + `available` / `missing_models` reporting
- `--disable-safety-checker`, fp16-only repo support (`variant="fp16"`)
- Cold-load 503 + `busy` / `queue` in `/health`, 400 validation on bad inputs
- Windows service (NSSM), firewall + Tailscale setup, test/panel scripts
