# Prism Model Test Results (2026-11-23)

All models tested end-to-end with real inputs and outputs saved as files.

## Image Generation (SD 1.5 / SDXL)

| Model | Status | Time | Device | Input Prompt | Output |
|-------|--------|------|--------|-------------|--------|
| DreamShaper_8 | PASS | 5.7s | cuda | cozy mountain cabin at sunset | [out_dreamshaper_8.png](file:///D:/AI/Prism/testdata/images/out_dreamshaper_8.png) |
| AbsoluteReality_1.8 | PASS | 5.2s | cuda | photorealistic portrait | [out_AbsoluteReality_1.8.png](file:///D:/AI/Prism/testdata/images/out_AbsoluteReality_1.8.png) |
| AnimagineXL_3.1 | PASS | 119.2s | cuda (offload) | anime girl masterpiece | [out_AnimagineXL_3.1.png](file:///D:/AI/Prism/testdata/images/out_AnimagineXL_3.1.png) |
| JuggernautXL_v9 | PASS | 136.4s | cuda (offload) | photorealistic portrait | [out_JuggernautXL_v9.png](file:///D:/AI/Prism/testdata/images/out_JuggernautXL_v9.png) |
| NoobAI-XL-Vpred | PASS | 37.1s | cuda (offload) | anime girl silver hair sunset | [out_NoobAI-XL-Vpred.png](file:///D:/AI/Prism/testdata/images/out_NoobAI-XL-Vpred.png) (better & faster than AnimagineXL) |
| amazing-logos-v3 | PASS | 6.6s | cuda | Simple elegant logo for Digital Art | [out_amazing-logos-v3.png](file:///D:/AI/Prism/testdata/images/out_amazing-logos-v3.png) (logo generation) |

## OCR (input: [ocr_handwritten_prescription.jpg](file:///D:/AI/Prism/testdata/images/ocr_handwritten_prescription.jpg))

All 4 models produced identical correct output (1278 chars, proper reading order).

| Model | Status | Time | Output |
|-------|--------|------|--------|
| nemotron-ocr-0 | PASS | 35.0s | [ocr_nemotron-ocr-0.txt](file:///D:/AI/Prism/testdata/images/ocr_nemotron-ocr-0.txt) (kept on disk - most reliable) |
| glm-ocr | PASS | 104.8s | (removed from disk, kept in code as downloadable) |
| qwen3-vl-2b | PASS | 30.2s | (kept on disk for VLM/caption/chat) |
| qianfan-ocr | PASS | 47.6s | (removed from disk, kept in code as downloadable) |

## Upscale / Restore

| Model | Status | Time | Input | Output |
|-------|--------|------|-------|--------|
| RealESRGAN_x4v3 | PASS | 16.8s | [dense_market_street.jpg](file:///D:/AI/Prism/testdata/images/dense_market_street.jpg) | [out_RealESRGAN_x4v3.png](file:///D:/AI/Prism/testdata/images/out_RealESRGAN_x4v3.png) (2560x1706) |
| swinir-realsr-x4 | PASS | 38.7s | [dense_market_street.jpg](file:///D:/AI/Prism/testdata/images/dense_market_street.jpg) | [out_swinir-realsr-x4.png](file:///D:/AI/Prism/testdata/images/out_swinir-realsr-x4.png) (2560x1706) (removed from disk, kept in code) |
| ddcolor-full | PASS | 16.2s | [dense_market_street.jpg](file:///D:/AI/Prism/testdata/images/dense_market_street.jpg) | [out_ddcolor-full.png](file:///D:/AI/Prism/testdata/images/out_ddcolor-full.png) (1280x853, color photo) |
| ddcolor-full (B&W) | PASS | 0.6s | [bw_market_street.jpg](file:///D:/AI/Prism/testdata/images/bw_market_street.jpg) | [out_ddcolor-bw.png](file:///D:/AI/Prism/testdata/images/out_ddcolor-bw.png) (1280x853, verified colorful) |

## ASR (input: testdata/audio/jfk.wav)

All models produced identical correct transcript.

| Model | Status | Time | Output | Kept on Disk |
|-------|--------|------|--------|-------------|
| turbo | PASS | 85.8s | [asr_turbo.txt](file:///D:/AI/Prism/testdata/images/asr_turbo.txt) | Yes (SK support) |
| large | PASS | 57.6s | [asr_large.txt](file:///D:/AI/Prism/testdata/images/asr_large.txt) | Yes (SK support) |
| granite-speech-470m | PASS | 49.4s | (removed from disk, kept in code) | No |
| nemotron-asr-0.6b | PASS | 41.2s | (removed from disk, kept in code) | No |

## Removed Models
- gfpgan-v1.4: Removed completely (from code and disk) - produced pink/orb image, no useful output
- ddcolor-tiny: Removed (redundant, ddcolor-full is better)
- instruct-pix2pix: Removed (model file didn't exist)
- swinir-realsr-x4: Removed from disk, kept in code as downloadable
- glm-ocr, qianfan-ocr: Removed from disk, kept in code as downloadable
- granite-speech-470m, nemotron-asr-0.6b: Removed from disk, kept in code as downloadable

## NSFW Safety Checker
- Enabled by default (enforced) - disable_safety_checker defaults to False
- Configurable via --disable-safety-checker CLI flag
