# Prism — Research: actionable improvements

Companion to `docs/ONDEVICE-PLAN.md`. Produced from the research prompt
in `~/Downloads/whisperapi deepresearch.txt`, scoped to the actual Prism
stack (PC + Android, ONNX LCM, whisper.cpp, NCNN, llama.cpp VLM, Ktor).
Evidence quality: web search + project docs; nothing was benchmarked here.

## Findings (sorted by Priority desc, Effort asc)

### 1. QNN / Hexagon NPU Execution Provider for the LCM UNet
- **Category:** 1 (ONNX runtime optimization)
- **What it is:** ONNX Runtime's QNN EP runs INT8 graphs on Snapdragon Hexagon NPUs instead of XNNPACK CPU. Qualcomm ships an ORT plugin EP (May 2026) and docs cover 16-bit-act / 8-bit-weight quantization.
- **Why it helps Prism:** the LCM UNet is INT8 QDQ and is the single biggest cost on Android (256 px / 4 steps ≈ 25 s). Snapdragon 8 Gen 2 targets Hexagon; even a 3–5× speedup turns 25 s into ~5–8 s.
- **Effort:** M (EP build + model re-quantization with QNN-compatible ops)
- **Risk:** high (QNN needs a QNN-compiled model, op coverage gaps, NPU differs per OEM)
- **Source:** onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- **Priority:** must-have

### 2. Drop NNAPI from any new code (deprecated Android 15)
- **Category:** 8 (memory/performance)
- **What it is:** NNAPI is deprecated in Android 15; Google directs new work to vendor delegates (QNN, CoreML, etc.) or LiteRT.
- **Why it helps Prism:** Prism already targets minSdk 29 / targetSdk 34 and exposes an "Auto/CPU/GPU (NNAPI)" setting. New NNAPI investment is dead-end; the setting should be relabeled to "Auto/CPU/GPU (NNAPI, deprecated)" and migrated to QNN/XNNPACK.
- **Effort:** S
- **Risk:** low
- **Source:** developer.android.com/ndk/guides/neuralnetworks/migration-guide
- **Priority:** must-have

### 3. Replace LCM-SD1.5 with MobileDiffusion / SnapGen (sub-second)
- **Category:** 2 (image quality/speed)
- **What it is:** Google's MobileDiffusion reaches sub-second 512×512 on mobile through a redesigned UNet (transformer+conv, fewer channels); SnapGen (CVPR 2025) ships a 379 M-param UNet.
- **Why it helps Prism:** LCM-SD1.5 QDQ is 824 MB and ~25 s at 256 px. A 300–400 M UNet at INT8 is ~4× smaller and an order of magnitude faster, and fits the "models never bundled, lazy download" rule.
- **Effort:** XL (new tokenizer/scheduler/VAE, retrain-or-convert weights)
- **Risk:** high (weight availability, license, quality on non-portrait prompts)
- **Source:** research.google/blog/mobilediffusion; arxiv 2311.16567
- **Priority:** research-only (no public converted weights today)

### 4. Evaluate Hyper-SD / SDXL-Lightning for the PC path
- **Category:** 2
- **What it is:** Distillation LoRAs/weights that reach 1–8 steps with quality close to 30-step SDXL (Hyper-SD supports FLUX.1-dev, SD3, SDXL, SD1.5).
- **Why it helps Prism:** the PC server already supports torch/diffusers; swapping the distilled checkpoint is a model-id change, no pipeline rewrite. Could cut PC generation 2–4×.
- **Effort:** S (model swap + prompt templates)
- **Risk:** medium (quality varies by prompt, license)
- **Source:** ByteDance/Hyper-SD (HuggingFace)
- **Priority:** nice-to-have

### 5. Sherpa-ONNX as an alternative Whisper backend (with Silero VAD)
- **Category:** 3 (audio/whisper)
- **What it is:** Sherpa-ONNX is an ONNX-Runtime-based speech stack with built-in VAD (Silero), streaming ASR, and Whisper models; used by several Android apps.
- **Why it helps Prism:** replaces the 30 s fixed chunking with VAD-driven segments (better boundaries, fewer empty chunks) and unifies the runtime with the image stack (one ORT instance).
- **Effort:** M (JNI/Gradle swap, VAD tuning)
- **Risk:** medium (quality regression on SK/CZ, new JNI surface)
- **Source:** github.com/k2-fsa/sherpa-onnx
- **Priority:** nice-to-have

### 6. Silero VAD before the existing whisper.cpp path
- **Category:** 3
- **What it is:** a tiny VAD model (~2 MB) that marks speech segments; feed only speech to whisper.cpp.
- **Why it helps Prism:** Prism already chunks at 30 s; VAD avoids transcribing silence and improves timestamp accuracy for SRT/VTT without changing the engine.
- **Effort:** S
- **Risk:** low
- **Source:** github.com/snakers4/silero-vad (ONNX)
- **Priority:** must-have (quick win)

### 7. Moondream2 / Phi-3-vision instead of SmolVLM for caption
- **Category:** 4 (VLM/caption)
- **What it is:** Moondream2 (~1.9 B) and Phi-3-vision (4.2 B) are small VLMs with strong captioning; Moondream has a GGUF/ONNX path and runs in ~1 GB.
- **Why it helps Prism:** the caption tab is still experimental (llama.cpp not bundled). Moondream2 is smaller than SmolVLM2-2.2B and has a cleaner mobile story; a lighter default reduces the 503 path.
- **Effort:** M (model swap + prompt format)
- **Risk:** medium (license, JNI unchanged)
- **Source:** moondream.ai; microsoft/Phi-3-vision
- **Priority:** nice-to-have

### 8. ONNX Runtime for the VLM (drop llama.cpp JNI)
- **Category:** 4
- **What it is:** run a VLM as ONNX (vision encoder + decoder with KV cache) instead of a separate CMake module.
- **Why it helps Prism:** removes the `:smolvlm` Gradle module, the `libsmolvlm_jni.so` build, and the `unsatisfied link` failure mode; one runtime for everything.
- **Effort:** XL (KV-cache loop, tokenizer, multimodal merge in Kotlin — same as the existing SmolVLM plan)
- **Risk:** high
- **Source:** onnxruntime.ai/docs/tutorials
- **Priority:** research-only

### 9. Switch upscale from NCNN to ONNX Runtime
- **Category:** 5 (upscale)
- **What it is:** Real-ESRGAN has community ONNX exports; ORT Mobile runs them with XNNPACK.
- **Why it helps Prism:** removes the ncnn fetch step, the Vulkan→CPU sticky fallback, and the 30 MB .so; one runtime, consistent memory handling.
- **Effort:** M (model + tiled inference)
- **Risk:** medium (quality parity with ncnn build)
- **Source:** onnxruntime.ai/docs/tutorials/mobile/superres.html
- **Priority:** nice-to-have

### 10. Real-ESRGAN-anime / SwinIR for photo vs. illustration
- **Category:** 5
- **What it is:** specialized SR models: anime6B for drawings, SwinIR/HAT for photos; smaller and sharper than x4plus on their domain.
- **Why it helps Prism:** the catalog already ships one upscale model; offering a 2-model picker (photo/anime) is a small catalog change with visible quality gains.
- **Effort:** S
- **Risk:** low
- **Source:** github.com/xinntao/Real-ESRGAN
- **Priority:** nice-to-have

### 11. MediaCodec with ByteBuffer input (drop EGL bridge for compress)
- **Category:** 6 (video)
- **What it is:** feed YUV/RGB frames through `MediaCodec` input buffers and use `Surface` only for the encoder's input, avoiding the EGL/GL bridge.
- **Why it helps Prism:** `VideoTranscoder.kt` currently re-encodes via MediaCodec + EGL; an EGL-free path is simpler, less fragile across OEMs, and avoids one GL context.
- **Effort:** M
- **Risk:** medium (color-format quirks per device)
- **Source:** developer.android.com/reference/android/media/MediaCodec
- **Priority:** nice-to-have

### 12. Hardware H.264/H.265 encode availability check
- **Category:** 6
- **What it is:** probe `MediaCodecList` for an encoder supporting the requested codec/profile/level before attempting transcode.
- **Why it helps Prism:** Prism already returns honest 503s for unsupported ops; an explicit capability probe turns "fails mid-run" into a clean 503 with the reason.
- **Effort:** S
- **Risk:** low
- **Source:** Android MediaCodec docs
- **Priority:** must-have (quick win)

### 13. Delta downloads for multi-file models
- **Category:** 7 (model management)
- **What it is:** HTTP Range + per-file SHA; only re-fetch files whose hash changed (LCM has 7 files).
- **Why it helps Prism:** `ModelDownloader` already resumes per file with Range and writes `.READY`. Adding a manifest-hash check on install would skip unchanged files on re-install/upgrade.
- **Effort:** S
- **Risk:** low
- **Source:** existing `ModelDownloader.kt` pattern
- **Priority:** nice-to-have

### 14. Automatic model selection by device tier
- **Category:** 7
- **What it is:** `HardwareProfile` already picks image size/threads; extend it to recommend a model id (tiny/base/small whisper, 256 vs 512 LCM).
- **Why it helps Prism:** prevents OOM on 4 GB devices and gives a one-tap "recommended" in the catalog.
- **Effort:** S
- **Risk:** low
- **Source:** existing `HardwareProfile`
- **Priority:** must-have (quick win)

### 15. Thermal-aware throttling
- **Category:** 8
- **What it is:** read `PowerManager` thermal status / `HardwarePropertiesManager` and reduce steps/size or pause when `THERMAL_STATUS_SEVERE`.
- **Why it helps Prism:** long 512 px / 6-step runs (~70 s) on a hot phone get killed by LMKD; throttling prevents the "process died" reports.
- **Effort:** S
- **Risk:** low
- **Source:** developer.android.com/reference/android/os/PowerManager
- **Priority:** nice-to-have

### 16. Latent + prompt cache
- **Category:** 9 (architecture)
- **What it is:** cache the CLIP text embedding for repeated prompts and the VAE latent for repeated source images (img2img/inpaint).
- **Why it helps Prism:** text encoder + VAE encode are ~20–30 % of a generation; caching cuts repeat-prompt latency and enables batch reuse.
- **Effort:** M
- **Risk:** low (memory cap required)
- **Source:** standard diffusion serving practice
- **Priority:** nice-to-have

### 17. SSE progress instead of polling
- **Category:** 9
- **What it is:** Server-Sent Events for `/v1/jobs/{id}` and download progress instead of client polling.
- **Why it helps Prism:** reduces wakeups and gives smoother progress bars; Ktor supports SSE via `ktor-server-sse`.
- **Effort:** M
- **Risk:** low
- **Source:** ktor.io/docs/server-sent-events.html
- **Priority:** nice-to-have

### 18. LAN auth token
- **Category:** 9
- **What it is:** optional bearer token on the Ktor server, configured in Settings, sent by clients.
- **Why it helps Prism:** the server binds 0.0.0.0 with no auth; any device on the LAN can generate. A token is a one-screen addition.
- **Effort:** S
- **Risk:** low
- **Source:** standard
- **Priority:** must-have (quick win for shared networks)

### 19. Determinism checks in the live suite
- **Category:** 10 (testing)
- **What it is:** fix the seed and assert bit-identical output across two runs for the same prompt.
- **Why it helps Prism:** catches provider/XNNPACK nondeterminism regressions in CI before they reach users.
- **Effort:** S
- **Risk:** low
- **Source:** existing `test_android_live.py`
- **Priority:** nice-to-have

### 20. Accessibility pass on the Android UI
- **Category:** 10
- **What it is:** content descriptions, focus order, TalkBack labels on the tab bar and Outputs dialog.
- **Why it helps Prism:** the UI is dense (tab bar, model manager, outputs list); a11y is currently unaudited.
- **Effort:** M
- **Risk:** low
- **Source:** Android Accessibility Scanner
- **Priority:** nice-to-have

### 21. Property-based fuzzing of the multipart parsers
- **Category:** 10
- **What it is:** fuzz the `/v1/images/*` multipart + JSON paths with malformed inputs.
- **Why it helps Prism:** Ktor multipart handling has had two bugs already (streamProvider vs provider); fuzzing catches the next one before release.
- **Effort:** S
- **Risk:** low
- **Source:** kotest-property / jqwik
- **Priority:** nice-to-have

### 22. Prompt cache for the text encoder (PC + Android)
- **Category:** 9
- **What it is:** LRU cache keyed by (prompt, model) → embedding tensor.
- **Why it helps Prism:** batch runs and chat reuse prompts; avoids re-tokenizing + re-encoding.
- **Effort:** S
- **Risk:** low
- **Source:** standard
- **Priority:** must-have (quick win)

## Quick wins (< 1 day)

1. Silero VAD before whisper.cpp (finding 6).
2. Hardware H.264/H.265 encode probe → honest 503 (finding 12).
3. Auto-select whisper/LCM model by device tier (finding 14).
4. LAN auth token (finding 18).
5. Text-embedding LRU cache (finding 22).
6. Relabel the NNAPI setting as deprecated (finding 2).
7. Add Moondream2 as an alternative caption model id (finding 7, catalog only).
8. Determinism check in `test_android_live.py` (finding 19).

## Not recommended

- **WebUI on Android** — explicitly ruled out by architecture; native only.
- **ffmpeg binary on Android** — GPL/licensing + 20–30 MB per ABI; MediaCodec + LAME covers the documented ops.
- **DirectML for the INT8 QDQ graph** — already evaluated; ORT DirectML cannot run the QDQ graph, CPU only.
- **NCNN Vulkan as the default upscale path** — returns a constant frame on several devices; keep as opt-in with sticky CPU fallback.
- **faster-whisper on Android** — whole-file decode before first segment; whisper.cpp with 30 s chunks is better.
- **Kotlin Multiplatform unification** — PC is Python; the cost of a shared KMP core exceeds the benefit given the divergent runtimes.
- **ConvInteger support in ORT Android** — still absent; QDQ conversion remains required.
- **Bundling models in the APK** — violates the lazy-download rule and bloats the APK; keep ModelDownloader.

## Notes on evidence

- QNN EP availability and the deprecation of NNAPI are documented by ONNX Runtime / Android NDK; performance numbers are indicative, not measured on Prism.
- MobileDiffusion/SnapGen figures come from the papers; converted ONNX weights are not publicly available for SD 1.5-class models today, so this is research-only.
- Sherpa-ONNX and Moondream2 are production-used in other apps but were not tested in Prism.