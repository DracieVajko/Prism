# Model ranking and test results

Source: `docs/MODEL-TESTS.md` (2026-11-19, RTX 3060 6 GB).

## Keep first (best per category)

|Rank|Model|Kind|Result|Time|Keep?|Output link|
|-:|-|-|-|-:|-|-|
|1|`turbo`|ASR|PASS|1.1 s|Keep|testdata/audio/jfk.wav|
|2|`glm-ocr`|OCR|PASS|50.5 s|Keep|testdata/images/test\_512.png|
|3|`RealESRGAN\_x4v3`|Upscale|PASS|22.9 s|Keep|pc/output/image/20261119-140226-req-8d1fcae1.png|
|4|`ddcolor-full`|Restoration|PASS|26.8 s|Keep|pc/output/image/20261119-140246-req-d30f1619.png|
|5|`DreamShaper\_8`|Image|PASS|312.4 s|Keep|pc/output/image/20261119-141059-test-DreamShaper\_8.png|

## Tested and passing

|Model|Kind|Time|Notes|Output|
|-|-|-:|-|-|
|`large`|ASR|21.9 s|correct|testdata/audio/jfk.wav|
|`granite-speech-470m`|ASR|199.0 s|correct|testdata/audio/jfk.wav|
|`swinir-realsr-x4`|Upscale|20.7 s|pass|pc/output/image/20261119-140434-req-de5e1221.png|
|`z-image-turbo`|Upscale|615.9 s|very slow|pc/output/image/20261119-085239-zimage-photo-2.png|
|`nemotron-ocr-0`|OCR|100.7 s|2,343 chars|pc/output/image/...|
|`granite-docling-258m`|OCR|10.8 s|short on slide|pc/output/image/...|
|`qwen3-vl-2b`|OCR/VLM|127.7 s / 19.5 s|OCR + describe|pc/output/image/...|
|`ovis-ocr2`|OCR|391.3 s|Markdown + boxes|pc/output/image/...|
|`paddleocr-vl-1.6`|OCR|89.2 s|pass|pc/output/image/...|
|`lightonocr-2-1b`|OCR|79.4 s|Markdown|pc/output/image/...|
|`qianfan-ocr`|OCR|6.9 min load|75 chars|pc/models/qianfan-ocr|
|`ddcolor-tiny`|Restoration|10.5 s|fallback only|pc/output/image/...|
|`gfpgan-v1.4`|Restoration|10.7 s|face restore|pc/output/image/...|
|`swinir-realsr-x4` restore|Restoration|62.1 s|pass|pc/output/image/...|
|`instruct-pix2pix`|Restoration|16.0 s|pass|pc/output/image/...|
|`AbsoluteReality\_1.8`|Image|281.1 s|512x512|pc/output/image/20261119-141540-test-AbsoluteReality\_18.png|
|`AnimagineXL\_3.1`|Image|156.8 s|512x512|pc/output/image/20261119-141817-test-AnimagineXL\_31.png|
|`JuggernautXL\_v9`|Image|157.8 s|512x512|pc/output/image/20261119-142055-test-JuggernautXL\_v9.png|

## Removed after testing

* `nemotron-asr-0.6b` (now fixed with Nemotron3\_5Asr\* classes)
* `qwen3-asr-0.6b`, `qwen3-asr-1.7b`, `qwencleo-asr-1.7b`
* `falcon-ocr`
* `unlimited-ocr-2`

## Download status

Optional models are downloading in the background. Results will be appended here after testing.

