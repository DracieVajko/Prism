# Model checklist — what is installed, what works, what to decide

Last inventory: 2026-11-11. Update this file when a model is downloaded,
removed, or its keep/delete decision changes.

## PC (`pc/models`)

Installed by default: **Whisper turbo + large only**. Every other Whisper size
is listed in the Models catalog as an optional download
(`WHISPER_SOURCES` in `pc/unified/models_manager.py`); nothing else is
downloaded automatically.

| Model | On disk | Used by | Status | Decision |
|-------|---------|---------|--------|----------|
| `faster-whisper-turbo` | 1.51 GB | `/v1/audio/transcriptions` (`model=turbo`) | Working | keep |
| `faster-whisper-large` | 2.88 GB | `model=large` | Working (slower) | keep, compare |
| `faster-whisper-tiny` | — | `model=tiny` | Optional download | keep optional |
| `faster-whisper-base` | — | `model=base` | Optional download | keep optional |
| `faster-whisper-small` | — | `model=small` | Optional download | keep optional |
| `faster-whisper-medium` | — | `model=medium` | Optional download | keep optional |
| `nemotron-asr-0.6b` | 3.07 GB | `model=nemotron-asr-0.6b` | Engine works (CUDA) | **TODO: keep or delete** |
| `SmolVLM_500M` | 0.95 GB | `/v1/images/caption`, vision chat | Working | keep |
| `LCM_Dreamshaper_ONNX` | 3.99 GB | `/v1/images/generations` | Working | keep |
| `DreamShaper_8` | 2.55 GB | image generations | Working | keep |
| `JuggernautXL_v9` | 6.62 GB | image generations | Working | keep |
| `AnimagineXL_3.1` | 6.46 GB | image generations | Working | keep |
| `AbsoluteReality_1.8` | 1.99 GB | image generations | Working | keep |
| `RealESRGAN_x4v3` | 0.12 GB | `/v1/images/upscale` | Working | keep |

Notes:

- `PRISM_LLM_MODEL` (default `gemma4:12b`) is only a **model name sent to an
  external provider** (Ollama/LM Studio). It does not load local weights.
  LM Studio currently serves `gemma-4-26B-A4B-it-QAT-GGUF`; set
  `PRISM_LLM_MODEL` to that ID when LM Studio is the provider.
- `pc/tools/download_gemma4.py` downloads `google/gemma-4-E2B-it` into
  `models/gemma-4-E2B-it`; that directory is **not present** on this machine.

## Android (device `10AEBU1ZU8001GK`, external files `.../files/models`)

| Model | On disk | Used by | Status | Decision |
|-------|---------|---------|--------|----------|
| `whisper-large-v3-turbo-q5_0` | 548 MB | `/v1/audio/transcriptions` | Working | keep |
| `lcm-dreamshaper-int8-onnx` | 2.3 GB | `/v1/images/generations` | Working | keep |
| `realesrgan-x4plus-ncnn` | 56 MB | `/v1/images/upscale` | Working | keep |
| `smolvlm-500m-gguf` | 521 MB | `/v1/images/caption`, `/v1/chat/completions` | Working | keep |
| `xiaomi-ocr-0` | 1.2 GB | `/v1/images/ocr` | Working | keep |
| `gemma4-e2b-gguf` | — not downloaded | Transcription post-processing switch (cleanup/translate/rewrite) | Catalog option only | **TODO: keep or remove from catalog** |
| Gemini Nano (AICore) | managed by the system | `/v1/text/summarize` | Working (verified 2026-11-11) | keep |

Notes:

- The Android Gemma 4 E2B entry is **optional** (3.0 GB, `gemma-4-E2B_q4_0-it.gguf`,
  llama.cpp text-only). Without it the post-processing switch logs
  "Gemma4 skipped: model not downloaded" and the transcript is unchanged.
- Gemini Nano is a separate on-device path and is the working summarizer today.

## Removed on 2026-11-11

- `gemma-4-12B-it` (22.31 GB, safetensors) — no Prism code path loaded it.
- `gemma-4-12B-it-q4` (6.63 GB, GGUF) — same; LM Studio has its own Gemma 4 26B.
- `faster-whisper-small` (0.45 GB) and `faster-whisper-tiny` (0.07 GB) — now
  optional downloads; turbo + large stay installed.
- Total freed: ~29.5 GB (`pc/models` 59.6 GB -> 30.1 GB).

## Decision queue

- [x] **Whisper sizes** — turbo + large installed; tiny/base/small/medium are
      optional downloads in the Models tab. Verified:
      `whisper_installed()` is true for turbo/large and false for the rest.
- [x] **Gemma 4 12B (PC)** — removed both downloads (2026-11-11).
- [ ] **Nemotron ASR 0.6B (PC, 3.07 GB)** — finish the comparison against
      Whisper turbo/large on the same Slovak sample, then decide keep/delete.
      The engine runs (CUDA, 40 languages, native punctuation), but there is
      no WER/quality number yet. Use
      `pc/scripts/compare_models/test_all_models.py` and store its results
      next to this file.
- [ ] **Gemma 4 E2B GGUF (Android, 3.0 GB)** — the catalog entry exists but the
      model is not downloaded and the post-processing switch is inert. Decide:
      keep the optional download and document it, or remove the entry so the UI
      does not offer a dead feature.

## Reported benchmark (from the earlier comparison run)

| Model | Reported size | Device | Language | Score | Notes |
|-------|---------------|--------|----------|-------|-------|
| Whisper Turbo | ~1.6 GB | CPU | en | 1.00 | working |
| Whisper Large | ~3.0 GB | CPU | en | 0.99 | partially working |
| Nemotron ASR 0.6B | ~2.5 GB | CUDA | auto | — | **not measured yet** |

These numbers are the user-reported run, not a fresh measurement. Re-run the
comparison before deleting Nemotron so the decision is based on the current
engine (the ASR facade and gain handling changed in November).
