# Prism Android — On-Device AI Models

## Available Models for Summarization / Cleanup

| Model | Size (quantized) | Backend | Device Support | Best For |
|-------|-----------------|---------|---------------|----------|
| **Gemini Nano** | ~1.8–3.25B (4-bit) | ML Kit GenAI API | Pixel 8+, Samsung S24+, Xiaomi, Motorola | Summarization, proofreading, rewrite (production-ready) |
| **Gemma 4 E2B** | ~3.1 GB (Q4_K_M) | LiteRT-LM | All arm64-v8a Android | Custom on-device summarization |
| **Gemma 4 E4B** | ~5.0 GB (Q4_K_M) | LiteRT-LM | All arm64-v8a Android | Higher quality summarization |
| **Bonsai 27B Q1_0** | ~3.9 GB (1-bit) | llama.cpp (custom) | All arm64-v8a | Extreme compression, 262K context |
| **Qwen 3.5 9B Distill** | ~5.8 GB (Q4_K_M) | llama.cpp / MNN | All arm64-v8a | Reasoning tasks |

---

## 1. Gemini Nano (ML Kit GenAI) — Recommended

### Dependency
```kotlin
implementation("com.google.mlkit:genai-prompt:1.0.0-beta1")
```

### Endpoint: `POST /v1/text/summarize`
Added to `ApiServer.kt`. Supports two modes:
- `summarize` — uses `Summarization.getClient()` with `SummarizerParams`
- `cleanup` — uses `Generation.getClient()` with `GenerationConfig`

### Device Support
Gemini Nano requires AICore system service and NPU support. Supported devices:
- Pixel 8, 8a, 9, 9 Pro
- Samsung S24, S25 series
- Xiaomi, Motorola (select models)

On unsupported devices, endpoint returns **503** with `engine: gemini-nano`.

### Kotlin Usage (direct ML Kit)
```kotlin
val summarizer = Summarization.getClient()
val params = SummarizerParams.builder()
    .setMaxSummaryLength(256)
    .build()
val summary = summarizer.summarize(text, params).await()
```

---

## 2. Gemma 4 via LiteRT-LM

Already integrated in Prism Android (`XiaomiOcrEngine.kt` uses LiteRT-LM). For summarization:

### Download Quantized Weights
```bash
# Gemma 4 E2B Q4_K_M (~3.1 GB)
huggingface-cli download unsloth/gemma-4-E2B-it-GGUF \
    gemma-4-E2B-it-Q4_K_M.gguf \
    --local-dir ./models/gemma-4-E2B-it-q4

# Gemma 4 E4B Q4_K_M (~5.0 GB)
huggingface-cli download unsloth/gemma-4-E4B-it-GGUF \
    gemma-4-E4B-it-Q4_K_M.gguf \
    --local-dir ./models/gemma-4-E4B-it-q4
```

### Context Window
Set context dynamically based on input length:
```kotlin
val contextSize = min(inputTokens * 2, 8192) // Cap at 8K for mobile
```

---

## 3. Bonsai 27B Q1_0 (Extreme Compression)

### Download
```bash
huggingface-cli download prism-ml/Bonsai-27B-gguf \
    Bonsai-27B-Q1_0.gguf \
    --local-dir ./models/bonsai-27b-q1
```

### Specs
- Size: ~3.9 GB (1-bit weights)
- Context: 262K tokens
- KV cache: Near-lossless 4-bit quantization
- Backend: llama.cpp (requires custom Android build)

### Trade-off
89.5% quality of full precision at 1/6th the size. Best for memory-constrained devices.

---

## 4. Qwen 3.5 9B Distill

### Download
```bash
huggingface-cli download unsloth/Qwen3.5-9B-GGUF \
    Qwen3.5-9B-Q4_K_M.gguf \
    --local-dir ./models/qwen-3.5-9b-q4
```

### Specs
- Size: ~5.8 GB (Q4_K_M)
- Context: 32K tokens
- Reasoning model (outputs thinking blocks)

---

## 5. LLM Optimization Techniques

See `docs/LLM-OPTIMIZATION-GUIDE.md` for complete guide including:
- Weight quantization (Q4_K_M, Q8_0, Q1_0)
- KV cache quantization
- Context window sizing (dynamic, not 256K by default)
- Lazy loading + idle unload
- Memory budgeting (at most one heavy model resident)

---

## 6. Files Modified

| File | Change |
|------|--------|
| `android/app/build.gradle.kts` | Added `com.google.mlkit:genai-prompt:1.0.0-beta1` |
| `android/app/src/main/java/.../ApiServer.kt` | Added `/v1/text/summarize` endpoint |
| `docs/LLM-OPTIMIZATION-GUIDE.md` | Created optimization guide |
| `pc/unified_server.py` | Default whisper model: small → turbo |

---

## 7. Next Steps

1. Wait for background tests to complete (whisper-turbo vs large vs nemotron vs gemma4-12B)
2. Wait for Gemma 4 12B Q4_K_M download to complete
3. Test summarization endpoint on Android device (when reconnected)
4. Benchmark all models on target device
5. Choose optimal model based on device RAM/NPU capabilities
