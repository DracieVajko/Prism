# Installed model test matrix

Date: 2026-11-19
Machine: RTX 3060 Laptop 6 GB, 32 GB RAM, Windows.
This run: LM Studio models unloaded (22 GB free RAM, GPU idle), Prism on
`127.0.0.1:8101` with `WHISPER_DEVICE=cuda`, `PRISM_OCR_DEVICE=auto`.

Inputs: the AI Income slide (OCR), `testdata/audio/jfk.wav` (ASR), a 384x512
photo (upscale/restoration/caption), a 256 px thumbnail (Z-Image).

| Model | Kind | Result | Time | Notes |
|---|---|---|---|---|
| `turbo` (Whisper) | audio | PASS | 1.1 s | 208 chars, correct JFK line |
| `large` (Whisper) | audio | PASS | 21.9 s | 207 chars, correct |
| `granite-speech-470m` | audio | PASS | 199.0 s | 208 chars, correct |
| `nemotron-asr-0.6b` | audio | FAIL | 54.0 s | HTTP 200 but **0 chars**; RNNT generate returns empty |
| `RealESRGAN_x4v3` | upscale | PASS | 22.9 s | 384x512 -> 768x1024 |
| `swinir-realsr-x4` | upscale | PASS | 20.7 s | 384x512 -> 768x1024 |
| `z-image-turbo` | upscale | PASS | 615.9 s | 256 -> 384x512, `zimage-turbo+realesrgan` |
| `nemotron-ocr-0` | ocr | PASS | 100.7 s | 2,343 chars on the slide |
| `granite-docling-258m` | ocr | PASS | 10.8 s | 44 chars (`<screenshot>`) on the slide; 4,987 chars on a PDF page earlier |
| `qwen3-vl-2b` | ocr | PASS | 127.7 s | 2,767 chars formatted OCR |
| `qwen3-vl-2b` describe | ocr | PASS | 19.5 s | 1,166-char image description |
| `ovis-ocr2` | ocr | PASS | 391.3 s | 7,995 chars Markdown with bounding boxes |
| `SmolVLM_500M` | vlm | PASS | 61.8 s | "In this image we can see a pen..." |
| `ddcolor-tiny` | restoration | PASS | 16.4 s | 384x512 colorized |
| `gfpgan-v1.4` | restoration | PASS | 10.7 s | 384x512 face restore |
| `swinir-realsr-x4` restore | restoration | PASS | 62.1 s | 384x512 -> 768x1024 |
| `instruct-pix2pix` | restoration | PASS | 16.0 s | after releasing the OCR model before loading |
| `DreamShaper_8` | image | PASS | 312.4 s | 512x512 |
| `AbsoluteReality_1.8` | image | PASS | 281.1 s | 512x512 |
| `AnimagineXL_3.1` | image | PASS | 156.8 s | 512x512 (requested 768; endpoint clamped) |
| `JuggernautXL_v9` | image | PASS | 157.8 s | 512x512 (requested 768; endpoint clamped) |

## Remaining failure

`nemotron-asr-0.6b` loads and the endpoint returns 200, but the RNNT
`generate()` produces an empty transcript. Earlier it raised
`'Nemotron3_5AsrForRNNT' object has no attribute 'get_audio_features'`; after
the model reloaded it now returns empty instead. The transformers 5.21 RNNT
generation path needs a fix (or a version pin) before this model is usable.
Whisper and Granite Speech cover the same audio tests successfully.

## Fixes that made this run pass

- `ocr_engine.py`: missing `os` import (broke every OCR load after the
  `PRISM_OCR_DEVICE` change); one OCR model resident at a time; release on
  reset; `transcribe()` delegates to `transcribe_pil()`.
- `/v1/images/restore`: releases SD/VLM/OCR/Z-Image engines first, so the
  2.6 GB InstructPix2Pix load fits next to nothing else.
- `download_models.py` / `models_manager.py`: `.incomplete/.part/.lock` no
  longer count as installed; the InstructPix2Pix download was completed
  (15.93 GB) and now loads.
- Test harness: ASR `text/plain` bodies are parsed correctly.

## Not tested

The optional catalog entries that are not installed (Qwen3-ASR, QwenCleo,
Cohere, Parakeet, NB-Whisper, ivrit, PaddleOCR-VL, GLM-OCR, FireRed-OCR,
Qianfan-OCR) were not downloaded; they route through the same engines but
need weights before they can be validated.
