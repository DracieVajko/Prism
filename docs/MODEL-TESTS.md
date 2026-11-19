# Installed model test matrix

Date: 2026-11-19
Machine: RTX 3060 Laptop 6 GB, 32 GB RAM, Windows.
Conditions: LM Studio LLMs resident (~20 GB RAM, GPU busy), Prism on `0.0.0.0:8101`
where noted. Models that need more memory than was free are marked BLOCKED.

| Model | Kind | Result | Time | Notes |
|---|---|---|---|---|
| `turbo` (Whisper) | audio | PASS | 1.1 s | 208 chars, correct JFK line |
| `large` (Whisper) | audio | PASS | 21.9 s | 207 chars, correct |
| `nemotron-asr-0.6b` | audio | FAIL | 291.6 s | `'Nemotron3_5AsrForRNNT' object has no attribute 'get_audio_features'` (transformers 5.21 incompatibility) |
| `granite-speech-470m` | audio | PASS | 199.0 s | 208 chars, correct |
| `RealESRGAN_x4v3` | upscale | PASS | 22.9 s | 384x512 -> 768x1024 |
| `swinir-realsr-x4` | upscale | PASS | 20.7 s | 384x512 -> 768x1024 |
| `z-image-turbo` | upscale | BLOCKED | 727 s | connection reset while loading 20 GB with LLMs resident; previously verified (1061.9 s cold, 31.0 s warm) |
| `nemotron-ocr-0` | ocr | PASS (prior) | 100 s | 2,343 chars on the slide |
| `granite-docling-258m` | ocr | PASS (prior) | 33-222 s | 44 chars (`<screenshot>`) on the slide; 4,987 chars on a PDF page |
| `qwen3-vl-2b` | ocr | PASS (prior) | 197 s | 2,767 chars formatted OCR |
| `qwen3-vl-2b` describe | ocr | PASS (prior) | 33 s | 1,166-char image description |
| `ovis-ocr2` | ocr | PASS (prior) | 666 s | 7,995 chars Markdown with bboxes; slow without flash-attention kernels |
| `SmolVLM_500M` | vlm | PASS | 61.8 s | "In this image we can see a pen..." |
| `ddcolor-tiny` | restoration | PASS | 16.4 s | 384x512 colorized |
| `gfpgan-v1.4` | restoration | PASS | 10.7 s | 384x512 face restore |
| `swinir-realsr-x4` restore | restoration | PASS | 62.1 s | 384x512 -> 768x1024 |
| `instruct-pix2pix` | restoration | FAIL | 0.4 s | only a 4.5 GB `.incomplete` file; the model was never finished |
| `DreamShaper_8` | image | PASS | 312.4 s | 512x512 |
| `AbsoluteReality_1.8` | image | PASS | 281.1 s | 512x512 |
| `AnimagineXL_3.1` | image | PASS | 156.8 s | 512x512 (requested 768; endpoint clamped) |
| `JuggernautXL_v9` | image | PASS | 157.8 s | 512x512 (requested 768; endpoint clamped) |

## Notes on the failures

- **Nemotron ASR**: the model and weights load, but `generate()` in transformers
  5.21 hits a missing `get_audio_features` in the RNNT generation mixin. This is
  a runtime incompatibility, not a missing download.
- **instruct-pix2pix**: the download stopped after 4.5 GB and left a `.cache`
  `.incomplete` file. Install detection now ignores `.incomplete/.part/.lock`,
  so it will no longer be reported as installed; re-download to finish it.
- **Z-Image**: needs ~20 GB of RAM for the weights. With the LLMs resident only
  ~2-3 GB was free, so the server connection was reset during the load. It
  works when the machine is free (verified earlier: 256 -> 512 px, 46 s warm).
- **OCR re-test**: the automated run failed on every OCR model because of a
  one-line bug introduced with `PRISM_OCR_DEVICE` (missing `import os`), not a
  model problem. That is fixed, but the retest could not load even the 0.5 GB
  Granite model with 1-2 GB free RAM; the PASS results above are from the
  earlier standalone runs.

## Code fixes made during this pass

- `ocr_engine.py`: added the missing `os` import; `transcribe()` now delegates
  to `transcribe_pil()` (one generation path); only one OCR model stays
  resident, releasing the previous one on switch.
- `download_models.py` / `models_manager.py`: partial downloads
  (`.incomplete`, `.part`, `.lock`) no longer count toward `installed`.
- Test harness: ASR responses are `text/plain`, so it now reads the body
  instead of assuming JSON.
