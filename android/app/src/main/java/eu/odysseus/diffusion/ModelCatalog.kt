package eu.odysseus.diffusion

/**
* Model catalog — the single source of truth for Android.
 *
* Rule: every URL is verified (HTTP 200 + Content-Length) and every model
* has an exact file list. Directory-style HF repos (e.g. LCM-Dreamshaper
* ONNX) are downloaded file by file, not as a "directory URL" — that was the
* reason the DreamShaper download failed before.
 *
* `engine` says what can actually run the model:
*  - whisper.cpp  → local transcription (JNI, works)
*  - onnxruntime  → local inference (engine in progress; weights downloadable)
*  - vulkan-ncnn  → Real-ESRGAN (weights downloadable; engine in progress)
*  - llama.cpp    → VLM runtime not bundled yet (experimental on-device)
 */
data class ModelFile(
    val path: String,
    val url: String,
    val sizeMb: Long,
    val sha256: String = "",
)

data class ModelAsset(
    val id: String,
    val kind: String, // audio | image | upscale | vlm | ocr | custom
    val label: String,
    val files: List<ModelFile>,
    val engine: String = "",
    val note: String = "",
    val custom: Boolean = false,
) {
    val sizeMb: Long get() = files.sumOf { it.sizeMb }

/** First URL — backward compatibility with the UI and API (`has_source`). */
    val url: String get() = files.firstOrNull()?.url ?: ""

/** SHA-256 of the first file (legacy API field). */
    val sha256: String get() = files.firstOrNull()?.sha256 ?: ""

    val hasSource: Boolean get() = files.isNotEmpty()

/** True if the model can run directly on the phone. */
    val runsOnDevice: Boolean get() = engine != "llama.cpp"
}

object ModelCatalog {
    private const val WHISPER = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

    private fun ggml(name: String, sizeMb: Long, sha: String = ""): ModelFile =
        ModelFile("ggml-$name.bin", "$WHISPER/ggml-$name.bin", sizeMb, sha)

/** Whisper (whisper.cpp GGML) — local transcription. */
    private val audio: List<ModelAsset> = listOf(
        ModelAsset(
            id = "whisper-large-v3-turbo-q5_0", kind = "audio", engine = "whisper.cpp",
            label = "Whisper Large v3 Turbo q5_0 — 547 MB, SK/CZ, fastest large model",
            files = listOf(ggml("large-v3-turbo-q5_0", 547)),
            note = "Default audio model. ~2.5x realtime on SD 8 Gen 2 (~2 h for 45 min).",
        ),
        ModelAsset(
            id = "silero-vad", kind = "audio", engine = "onnxruntime",
            label = "Silero VAD — 2 MB, voice activity detection",
            files = listOf(
                ModelFile("silero_vad.onnx",
                    "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx", 2),
            ),
            note = "Voice activity detection. Pre-filter audio before transcription to skip silence.",
        ),
    )

    /**
* LCM-Dreamshaper v7 INT8 ONNX (the repo is a directory, hence the explicit
* file list). The UNet graph is small; weights live in an external .data file —
* both must be downloaded.
     */
    private val image: List<ModelAsset> = listOf(
        ModelAsset(
            id = "lcm-dreamshaper-int8-onnx", kind = "image", engine = "onnxruntime",
            label = "LCM-Dreamshaper v7 INT8 ONNX — 1.6 GB, 4 steps",
            files = listOf(
                ModelFile("unet/model.onnx",
                    "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/unet/model.onnx", 3),
                ModelFile("unet/model_int8.onnx.data",
                    "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/unet/model_int8.onnx.data", 821),
                ModelFile("text_encoder/model.onnx",
                    "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/text_encoder/model.onnx", 470),
                ModelFile("vae_decoder/model.onnx",
                    "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/vae_decoder/model.onnx", 189),
                ModelFile("vae_encoder/model.onnx",
                    "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/vae_encoder/model.onnx", 130),
                ModelFile("tokenizer/vocab.json",
                    "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/tokenizer/vocab.json", 1),
                ModelFile("tokenizer/merges.txt",
                    "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/tokenizer/merges.txt", 1),
            ),
            note = "JSON graph + external weights; engine in progress. Use PC (:8101) for NVIDIA.",
        ),
        ModelAsset(
            id = "hyper-sd15-1step-lora", kind = "image", engine = "onnxruntime",
            label = "Hyper-SD 1-Step LoRA — 269 MB, 1-step generation",
            files = listOf(
                ModelFile("Hyper-SD15-1step-lora.safetensors",
                    "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD15-1step-lora.safetensors", 269),
            ),
            note = "ByteDance Hyper-SD 1-step LoRA. Requires LCM-Dreamshaper base + TCDScheduler. ~7s generation.",
        ),
    )

    private val upscale: List<ModelAsset> = listOf(
        ModelAsset(
            id = "realesrgan-x4plus-ncnn", kind = "upscale", engine = "vulkan-ncnn",
            label = "Real-ESRGAN x4plus NCNN — 45 MB",
            files = listOf(
                ModelFile("realesrgan-x4plus-ncnn.zip",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip", 45),
            ),
            note = "Zip contains .param/.bin weights; auto-extracted.",
        ),
        ModelAsset(
            id = "realesrgan-x4plus-anime-6b", kind = "upscale", engine = "onnxruntime",
            label = "Real-ESRGAN x4plus Anime 6B — 17.9 MB, 6-block, faster on CPU",
            files = listOf(
                ModelFile("RealESRGAN_x4plus_anime_6B.onnx",
                    "https://huggingface.co/deepghs/imgutils-models/resolve/main/real_esrgan/RealESRGAN_x4plus_anime_6B.onnx", 17,
                    "2648cab4c4343541c1aa291c6754e9e8edbe7a813fffc2a677423dd12cb6b7f7"),
            ),
            note = "6-block RRDBNet, ~4x faster on CPU than x4plus. Optimized for anime/stylized images.",
        ),
    )

/** VLM: runtime is llama.cpp (libmtmd) — not bundled yet; experimental. */
    private val vlm: List<ModelAsset> = listOf(
        ModelAsset(
            id = "gemma4-e2b-gguf", kind = "vlm", engine = "llama.cpp",
            label = "Gemma4 E2B Instruct Q4_0 (text) - 3.0 GB",
            files = listOf(
                ModelFile("gemma-4-E2B_q4_0-it.gguf",
                    "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf", 3042),
            ),
            note = "Gemma4 E2B for text post-processing (cleanup, translate, rewrite). Text-only, no vision.",
        ),
        ModelAsset(
            id = "smolvlm-500m-gguf", kind = "vlm", engine = "llama.cpp",
            label = "SmolVLM 500M Instruct Q8_0 + vision mmproj — 520 MB",
            files = listOf(
                ModelFile("SmolVLM-500M-Instruct-Q8_0.gguf",
                    "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf", 417),
                ModelFile("mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
                    "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf", 104),
            ),
            note = "On-device caption/chat needs the llama.cpp VLM runtime (libsmolvlm_jni), not bundled yet.",
        ),
        ModelAsset(
            id = "smolvlm2-500m-gguf", kind = "vlm", engine = "llama.cpp",
            label = "SmolVLM2 500M Video Instruct Q8_0 + mmproj — 520 MB",
            files = listOf(
                ModelFile("SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
                    "https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF/resolve/main/SmolVLM2-500M-Video-Instruct-Q8_0.gguf", 417),
                ModelFile("mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
                    "https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf", 104),
            ),
            note = "On-device caption/chat needs the llama.cpp VLM runtime (libsmolvlm_jni), not bundled yet.",
        ),
        ModelAsset(
            id = "moondream2-text-v1", kind = "vlm", engine = "llama.cpp",
            label = "Moondream2 — 1.9B VLM, fast captioning",
            files = listOf(
                ModelFile("moondream2-text-model-f16.gguf",
                    "https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-text-model-f16.gguf", 417),
            ),
            note = "Small VLM (~1.9B params) for fast image captioning. GGUF format for llama.cpp runtime.",
        ),
    )

/** OCR: Xiaomi-OCR-0 as the official LiteRT-LM conversion (com.google.ai.edge.litertlm). */
private val ocr: List<ModelAsset> = listOf(
    ModelAsset(
        id = "xiaomi-ocr-0", kind = "ocr", engine = "litertlm",
        label = "Xiaomi-OCR-0 int8 LiteRT-LM — 1.25 GB, OCR / tables / key info",
        files = listOf(
            ModelFile("Xiaomi-OCR-0_int8.litertlm",
                "https://huggingface.co/litert-community/Xiaomi-OCR-0/resolve/main/Xiaomi-OCR-0_int8.litertlm",
                1254, "82880f7361b90f51a3fda9609878e24b4a00e95a058f8ff47391b90e5e95d9be"),
        ),
        note = "Official Google conversion (litert-community, min runtime 0.19.0). " +
               "Runs on CPU: ~10-60 s per page, peak RAM ~3-4 GB — the server " +
               "refuses OCR with an honest 503 when memory is short.",
    ),
)

    val assets: List<ModelAsset> = audio + image + upscale + vlm + ocr

    val kinds: List<String> = listOf("audio", "image", "upscale", "vlm", "ocr", "custom")

    fun kindLabel(kind: String): String = when (kind) {
        "audio" -> "Transcription (Whisper)"
        "image" -> "Image generation"
        "upscale" -> "Upscale"
        "vlm" -> "Vision-language (VLM)"
        "ocr" -> "OCR (Xiaomi-OCR-0)"
        "custom" -> "Custom (own URL)"
        else -> kind
    }

    fun find(id: String): ModelAsset? = assets.firstOrNull { it.id == id }

    fun forImageModel(registryId: String): ModelAsset =
        assets.firstOrNull { it.kind == "image" && it.id == registryId }
            ?: assets.first { it.kind == "image" }

    fun forKind(kind: String): List<ModelAsset> = assets.filter { it.kind == kind }
}
