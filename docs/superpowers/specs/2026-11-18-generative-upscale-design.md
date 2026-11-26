# Generative and deterministic upscale engines (design)

Date: 2026-11-18
Status: approved (approach C chosen by the user)
Scope: PC product only (`pc/`); Android unchanged.

## Intent

Real-ESRGAN is the only upscale engine in Lenswright today. It is fast and
deterministic, but it cannot rebuild detail that the source does not contain.
The user wants two optional engines beside it:

1. a stronger deterministic upscaler that still behaves like ESRGAN
   (seconds per image, no invented content);
2. a generative refine pass that can rebuild texture and small detail when
   quality matters more than speed.

The default must not change. Existing callers that send no `model` keep
getting Real-ESRGAN with identical output.

## Goals

- `/v1/images/upscale` accepts `model` values:
  `RealESRGAN_x4v3` (default), `swinir-realsr-x4` (deterministic),
  `z-image-turbo` (generative refine).
- Missing weights return 503 with an actionable hint, never a fake success.
- The WebUI Upscale tab lists the engines through the existing model catalog.
- Engine choice is recorded in the response (`engine` field).

## Non-goals

- MiniMax-H3-Image. Its community licence excludes the EU (the user is in
  Slovakia), its 6.4 GB figure is only the Q2_K DiT, and the full stack needs
  ~20 GB (12-27 GB text encoder + 1.17 GB VAE). Rejected for licence and size.
- Android on-device generative upscale. A 6B model cannot run on the phone.
- Video upscaling, tiling UI, batch prompt editing.

## Constraints

- RTX 3060 Laptop, 6 GB VRAM, 32 GB RAM, Windows.
- diffusers 0.42.0 ships `ZImagePipeline` and `ZImageImg2ImgPipeline`.
- Models are downloaded lazily into `pc/models/`; weights are never bundled.
- One engine resident at a time (`ModelManager` gate).

## Approaches considered

- **A. Deterministic engines only.** Add SwinIR (already vendored as ONNX) and
  later HAT behind the `model` selector. Cheap, fast, 4 GB-safe.
- **B. Generative refine only.** ESRGAN base pass + Z-Image Turbo img2img.
  Best quality, but minutes per image on 6 GB and needs careful tiling.
- **C. Both, A first, B behind the same selector.** Chosen.

## Decisions

### SwinIR (deterministic)

- Reuse `unified/swinir.py` (ONNX, Apache-2.0, 64 px tiles, 8 px overlap).
- It is x4 only; scale 2/3 is produced by downscaling the x4 result, the same
  rule Real-ESRGAN already follows.
- Raise the input cap from 512 px to 1024 px (`LENSWRIGHT_SWINIR_MAX_SIDE`
  override). A 1024 px input produces 4096 px output; the endpoint's 8192 px
  cap still applies.
- Report `engine = "swinir-realsr-x4"`.

### Z-Image Turbo (generative refine)

- Pipeline: `ZImageImg2ImgPipeline`, bfloat16. Below 9 GB VRAM it uses
  `enable_sequential_cpu_offload()` (the bf16 transformer alone is 12.3 GB and
  does not fit a 6 GB card as a whole module); larger cards get
  `enable_model_cpu_offload()`. VAE slicing/tiling and attention slicing on.
- Chain: Real-ESRGAN x4 to the requested scale, then Z-Image img2img at
  `strength=0.33` (ComfyUI's 2K-upscaler default), `num_inference_steps=9`,
  `guidance_scale=0.0` (Turbo is CFG-distilled).
- Images above 1024 px are refined in 1024 px tiles with a 96 px feathered
  overlap. More than 9 tiles is refused with a clear error rather than silently
  shrinking the result; the base pass keeps the existing 8192 px cap.
- The prompt is optional; an empty prompt uses a neutral "high quality,
  sharp natural detail" instruction.
- Report `engine = "zimage-turbo+realesrgan"`. On failure the base ESRGAN
  result is returned with `engine = "realesrgan-torch"` and a warning, so a
  request never loses the deterministic result.
- Weights: `pc/models/z-image-turbo` (Apache-2.0, ~33 GB). Download through
  `download_models.py --preset zimage` or the WebUI Download button.

### API

`UpscaleRequest` gains `prompt`, `strength`, `seed`. They are ignored by the
ESRGAN and SwinIR paths.

## Risks

- Generative upscale is slow on 6 GB: expect tens of seconds to a few minutes
  per image with CPU offload, versus seconds for ESRGAN/SwinIR.
- Img2img re-draws detail; faces, text and small objects can change. The
  default strength stays conservative (0.33) and the mode is optional.
- The 33 GB download is large; the catalog shows it as a separate entry.

## Verification

- Unit: engine dispatch by `model`, scale 2/3/4 maths, missing-model errors.
- Smoke: SwinIR on a synthetic image (ONNX session), Z-Image on a 512 px
  image once weights exist.
- Endpoint: `/v1/images/upscale` for each engine, plus an unknown model name.
- Measured so far: SwinIR 64 -> 128 px in 6.4 s cold (endpoint), and
  512 -> 2048 px in ~2.5 min on the RTX 3060. That is minutes, not seconds;
  Real-ESRGAN remains the fast default.
- Note: the Hub connection is cut after ~150-200 MB, so the 33 GB Z-Image
  snapshot was fetched in ranged chunks; every file matches its SHA-256.
- End-to-end: 256 -> 512 px returns 200 with `engine=zimage-turbo+realesrgan`.
  Cold call 19.6 min (20 GB load on a 32 GB machine), warm call 46 s at
  strength 0.33. The 900 s idle unload releases the pipeline, so a later
  session pays the load again.
