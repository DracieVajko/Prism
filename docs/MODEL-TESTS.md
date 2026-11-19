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
| `RealESRGAN_x4v3` | upscale | PASS | 22.9 s | 384x512 -> 768x1024 |
| `swinir-realsr-x4` | upscale | PASS | 20.7 s | 384x512 -> 768x1024 |
| `z-image-turbo` | upscale | PASS | 615.9 s | 256 -> 384x512, `zimage-turbo+realesrgan` |
| `nemotron-ocr-0` | ocr | PASS | 100.7 s | 2,343 chars on the slide |
| `granite-docling-258m` | ocr | PASS | 10.8 s | 44 chars (`<screenshot>`) on the slide; 4,987 chars on a PDF page earlier |
| `qwen3-vl-2b` | ocr | PASS | 127.7 s | 2,767 chars formatted OCR |
| `qwen3-vl-2b` describe | ocr | PASS | 19.5 s | 1,166-char image description |
| `ovis-ocr2` | ocr | PASS | 391.3 s | 7,995 chars Markdown with bounding boxes |
| `glm-ocr` | ocr | PASS | 50.5 s | 2,602 chars, clean plain text; 205.3 s through the API |
| `paddleocr-vl-1.6` | ocr | PASS | 89.2 s | 2,526 chars; 157.3 s through the API |
| `lightonocr-2-1b` | ocr | PASS | 79.4 s | 2,974 chars Markdown; 262.6 s through the API |
| `qianfan-ocr` | ocr | PENDING | 272.8 s load | 9.5 GB downloaded; 4-bit CPU load verified, full bf16 inference blocked by an active 12B LM Studio model |
| `SmolVLM_500M` | vlm | PASS | 61.8 s | "In this image we can see a pen..." |
| `ddcolor-tiny` | restoration | PASS | 16.4 s | 384x512 colorized |
| `gfpgan-v1.4` | restoration | PASS | 10.7 s | 384x512 face restore |
| `swinir-realsr-x4` restore | restoration | PASS | 62.1 s | 384x512 -> 768x1024 |
| `instruct-pix2pix` | restoration | PASS | 16.0 s | after releasing the OCR model before loading |
| `DreamShaper_8` | image | PASS | 312.4 s | 512x512 |
| `AbsoluteReality_1.8` | image | PASS | 281.1 s | 512x512 |
| `AnimagineXL_3.1` | image | PASS | 156.8 s | 512x512 (requested 768; endpoint clamped) |
| `JuggernautXL_v9` | image | PASS | 157.8 s | 512x512 (requested 768; endpoint clamped) |

## Removed after testing

- `nemotron-asr-0.6b` (and the 30B variants): loaded but returned an empty
  transcript via the transformers 5.21 RNNT path; the 30B models do not fit.
- `qwen3-asr-0.6b`, `qwen3-asr-1.7b`, `qwencleo-asr-1.7b`: the `qwen-asr`
  wrapper requires huggingface-hub < 1.0, which breaks transformers 5.21 and
  diffusers 0.42; raw `generate()` also mismatched the checkpoint.
- `falcon-ocr`: Triton kernel needs a CUDA C compiler (MSVC); the Windows host
  has none, and the eager fallback hit a CUDA device-side assert.
- `unlimited-ocr-2`: custom `UnlimitedOCRForCausalLM` tested upstream on
  transformers 4.52/Python 3.10; its Triton sparse kernels need a CUDA
  toolchain and the stack here is transformers 5.21 on Windows.

Their model directories may remain on disk, but they are no longer offered in
the catalog. Whisper, Granite Speech and the remaining OCR models cover those
languages/features.

## Fixes that made this run pass

- `ocr_engine.py`: missing `os` import (broke every OCR load after the
  `PRISM_OCR_DEVICE` change); one OCR model resident at a time; release on
  reset; `transcribe()` delegates to `transcribe_pil()`.
- `ocr_engine.py`: added the `native-vl` family. GLM-OCR, PaddleOCR-VL,
  LightOnOCR 2 and Qianfan-OCR take the PIL image inside the chat message and
  tokenise it themselves; the old Qwen `images=[image]` call lost the image
  token for some processors.
- `ocr_engine.py`: LightOnOCR 2 is loaded with its own
  `LightOnOcrForConditionalGeneration` / `LightOnOcrProcessor` classes. Its
  config says `model_type: mistral3`, so `AutoModelForImageTextToText` built a
  Mistral3 module and dropped the vision weights (0 chars).
- PaddleOCR-VL is loaded through the native transformers class with
  `trust_remote_code=False`; the remote-code path is the one that wanted
  `rope_parameters`.
- `/v1/images/restore`: releases SD/VLM/OCR/Z-Image engines first, so the
  2.6 GB InstructPix2Pix load fits next to nothing else.
- `download_models.py` / `models_manager.py`: `.incomplete/.part/.lock` no
  longer count as installed; the InstructPix2Pix download was completed
  (15.93 GB) and now loads.
- Test harness: ASR `text/plain` bodies are parsed correctly.

## Not tested

The optional catalog entries that are not installed (Cohere, Parakeet,
NB-Whisper, ivrit, FireRed-OCR, Qianfan-OCR) route through the same engines but
need weights before they can be validated. Qianfan-OCR is downloaded; its run
was blocked by the 6 GB GPU plus an active 12B LM Studio model (only 5 GB RAM
free) and must be retried with the GPU idle.
