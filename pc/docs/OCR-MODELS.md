# OCR / document / image-understanding models

Select with `model=` on `POST /v1/images/ocr` (multipart: `file`, `model`,
`language`, `task=ocr|describe`). `model=auto` (the default) prefers
`glm-ocr` when installed because it preserves UI/section reading order, then
falls back to `nemotron-ocr-0`. Pass `model=nemotron-ocr-0` explicitly for the
fastest plain line-by-line text. `task=describe` needs a general VLM
(`qwen3-vl-2b`); document-only parsers reject it with a 400.

| Model id | Size | Strength | Licence | Status |
|---|---|---|---|---|
| `nemotron-ocr-0` | 1.7 GB | fast plain-text OCR, multilingual | Apache-2.0 | tested (2,343 chars on the slide; 100-163 s) |
| `granite-docling-258m` | 0.5 GB | page/PDF -> DocTags/Markdown, tables, formulas | Apache-2.0 | tested (4,987 chars on a PDF page) |
| `qwen3-vl-2b` | 4.3 GB | 32-language OCR + objects/description | Apache-2.0 | tested (2,767 chars OCR, 1,166-char description) |
| `ovis-ocr2` | 1.7 GB | 853M, 96.58 OmniDocBench v1.6, layout/tables/formulas | Apache-2.0 | tested (7,995 chars with bbox Markdown; 666 s - slow) |
| `glm-ocr` | 2.7 GB | 0.9B, 94.62 OmniDocBench, tables/formulas, 8 languages | MIT | tested (2,602 chars, 50.5 s direct / 205.3 s API) |
| `paddleocr-vl-1.6` | 1.9 GB | 0.96B, 109 languages, layouts/tables/formulas | Apache-2.0 | tested (2,526 chars, 89.2 s direct / 157.3 s API) |
| `lightonocr-2-1b` | 2.0 GB | 1B, fast end-to-end page OCR, Markdown | Apache-2.0 | tested (2,974 chars Markdown, 79.4 s direct / 262.6 s API) |
| `qianfan-ocr` | 9.5 GB | 4.7B, 192 languages, KIE/layout, Markdown | Apache-2.0 | downloaded; 4-bit load verified, full bf16 run pending (needs GPU idle) |
| `firered-ocr` | 4.3 GB | Qwen3-VL-2B fine-tune for document parsing | Apache-2.0 | wired, not tested |

## Not wired, and why

| Model | Size | Reason |
|---|---|---|
| `baidu/Unlimited-OCR-2` | 6.7 GB | Custom `UnlimitedOCRForCausalLM`; upstream tested on transformers 4.52/Python 3.10 and its Triton sparse kernels need a CUDA toolchain. Not compatible with this host stack (transformers 5.21, Windows, no MSVC). |
| `baidu/Unlimited-OCR` | 13.4 GB | same, plus far over the VRAM/RAM budget |
| `tiiuae/Falcon-OCR` | 1.1 GB | Remote code requires Triton; on Windows without MSVC the kernel fails with `Failed to find C compiler`, and the eager fallback triggered a CUDA device-side assert. |
| `dots.ocr` | 6.1 GB | `DotsOCRForCausalLM`, custom runtime; over 6 GB |
| `deepseek-ai/DeepSeek-OCR-2` | 6.8 GB | custom code; over 6 GB |
| `datalab-to/surya-ocr-2` | 1.4 GB | separate `surya` package/runtime, not a plain transformers pipeline |
| `datalab-to/chandra-ocr-2` | 10.6 GB | separate `chandra` package/runtime; OpenRAIL threshold |
| `allenai/olmOCR-2-7B` | 16.6 GB | too large for this machine |
| `datalab-to/chandra` | 10.6 GB | too large; OpenRAIL threshold |
| `Surya OCR 2` | 1.4 GB | separate `surya` runtime |
| `PaddleOCR PP-OCRv6` | 34.5M | classic detection/recognition pipeline, not a VLM |
| `Jina OCR v1` | 6.7 GB | CC BY-NC 4.0, non-commercial |
| `Underdog Bark` | 0.4 GB | CC BY-NC 4.0, non-commercial |

## Download and use

```powershell
python pc\download_models.py glm-ocr
python pc\download_models.py paddleocr-vl-1.6
python pc\download_models.py lightonocr-2-1b
python pc\download_models.py qianfan-ocr
```

```bash
# OCR
curl -F file=@page.png -F model=ovis-ocr2 http://127.0.0.1:8101/v1/images/ocr
# description (general VLM only)
curl -F file=@photo.jpg -F model=qwen3-vl-2b -F task=describe \
     http://127.0.0.1:8101/v1/images/ocr
```

All models appear in `GET /v1/models/catalog` with kind `ocr`, so the WebUI
Models tab can download them. They are PC-only; Android's OCR is a separate
LiteRT engine.

## Practical notes

- On a 6 GB card, `qwen3-vl-2b` and `firered-ocr` overflow and fall back to CPU
  layers; `ovis-ocr2`, `glm-ocr`, `paddleocr-vl-1.6`, `lightonocr-2-1b` and
  `granite-docling` fit comfortably.
- `LENSWRIGHT_OCR_DEVICE=auto|cpu|cuda` controls placement. Use `cpu` when the GPU
  is busy with an LLM; `qianfan-ocr` (9.5 GB) and other large models only load
  with offload.
- LightOnOCR 2 declares `model_type: mistral3`, so the generic auto class builds
  the wrong module and drops the vision weights. The engine loads its dedicated
  `LightOnOcrForConditionalGeneration` / `LightOnOcrProcessor` classes instead.
- PaddleOCR-VL 1.6 has native transformers classes. Load it with
  `trust_remote_code=False`; the repo's remote code path is the one that wanted
  a `rope_parameters` config transformers 5.21 does not provide.
- On the tall AI-income slide, LightOnOCR produced the cleanest Markdown
  (2,974 chars), GLM-OCR the cleanest plain text (2,602), PaddleOCR-VL lost a
  few lines in the marketing block (2,526), and Nemotron was fastest/plainest
  (2,343). This is one sample, not a benchmark.
- GLM-OCR is the only model here that preserved the UI structure on that slide:
  section headings, reading order and grouped cards. Nemotron is a line-level
  OCR engine, so its output is accurate but unformatted. `auto` therefore picks
  GLM when it is installed; use `model=nemotron-ocr-0` when speed matters more
  than structure.
- OvisOCR2 is the accuracy leader per parameter, but its Qwen3.5 hybrid layers
  want `flash-linear-attention` and `causal_conv1d`; without them it took
  666 s for one image here. Use `nemotron-ocr-0` or `granite-docling-258m`
  for interactive work and Ovis for offline accuracy runs.
- Document parsers return Markdown/DocTags with layout when the model supports
  it; `qwen3-vl-2b` is the only general image-description model here.
- Render PDFs at 200-300 DPI before OCR; that matters more than the model
  choice on scanned pages.
