# Prism LLM Model Optimization Guide

## Overview

This guide covers optimization techniques for LLM models used in Prism PC server and Android app, including Gemma 4, Whisper, Nemotron ASR, and on-device Android models.

---

## 1. Model Weight Quantization

### Quantization Options by Model Family

| Model | Full Precision | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_M | Q2_K | Q1_0 |
|-------|---------------|------|------|--------|--------|--------|------|------|
| Gemma 4 E2B (2.3B) | ~9.3 GB | ~5.1 GB | ~4.2 GB | ~3.7 GB | ~3.1 GB | ~2.6 GB | ~2.2 GB | — |
| Gemma 4 E4B (4.5B) | ~15.1 GB | ~8.2 GB | ~6.8 GB | ~6.0 GB | ~5.0 GB | ~4.2 GB | ~3.5 GB | — |
| Gemma 4 12B | ~24 GB | ~13 GB | ~11 GB | ~9.5 GB | ~7.4 GB | ~6.2 GB | — | — |
| Bonsai 27B | — | — | — | — | — | — | — | ~3.9 GB (1-bit) |
| Qwen 3.5 9B | ~18 GB | ~9.8 GB | ~7.6 GB | ~6.6 GB | ~5.8 GB | — | — | — |

### Recommendations by Use Case

- **PC server (33.7 GB RAM, RTX 3060 6 GB VRAM)**: Q4_K_M or Q5_K_M for CPU inference; Q8_0 if VRAM allows
- **Android (on-device)**: Q4_K_M or Q3_K_M for memory-constrained devices; Q1_0 for Bonsai 27B extreme compression
- **Mobile NPU**: INT8 quantized via LiteRT-LM

---

## 2. KV Cache Quantization

The KV cache stores intermediate attention states during generation. For long contexts, this can consume significant memory.

### KV Cache Quantization Levels

| Method | Memory Reduction | Quality Impact | Best For |
|--------|-----------------|----------------|----------|
| FP16 (baseline) | 1× | None | Short context, high quality |
| Q8_0 | ~2× | Negligible | General purpose |
| Q6_K | ~2.7× | Minimal | Medium context |
| Q4_K | ~4× | Slight | Long context, mobile |
| Q4_1 (Bonsai drafter) | ~4× | Minimal | Speculative decoding |

### Implementation

```python
# PC server - unified_server.py
# KV cache quantization via faster-whisper VAD and segment-based processing
# Whisper uses 30s chunks with VAD filter - no KV cache needed (non-autoregressive)

# Gemma4 post-processing - bounded generation
budget = max(48, min(1024, prompt_tokens))  # Limit output tokens
# This bounds KV cache growth

# Android - LiteRT-LM KV cache
# LiteRT-LM manages KV cache internally; set context window only as needed
```

### Context Window Optimization

**Do NOT load 256K context by default.** Set context only as needed:

```python
# Bad: loading maximum context always
context_size = 262144  # Wastes memory for short inputs

# Good: dynamic context sizing
def get_optimal_context(input_tokens: int) -> int:
    """Set context window to 1.5x input length, capped at model max."""
    needed = input_tokens * 2  # Input + output buffer
    return min(needed, 8192)  # Cap at 8K for most on-device use cases
```

---

## 3. Optimal Model Loading

### Lazy Loading Pattern

Models should NEVER load at startup. Load on first request:

```python
# PC server pattern (already implemented in Prism)
_model = None

def init():
    global _model
    if _model is not None:
        return True
    _model = load_model()  # Load only when needed
    return True

def release():
    global _model
    _model = None
    clear_cache()
```

### Idle Unload

Unload models after idle period to free memory:

```python
IDLE_UNLOAD_SECONDS = 600  # 10 minutes

# PC server - Audio engine unloads on idle/switch
# Android - Engine per OCR request, serialize via heavyDispatcher
```

### Memory Budgeting

| Component | Memory Budget |
|-----------|--------------|
| Whisper turbo | ~1.6 GB |
| Nemotron 0.6B | ~2.5 GB VRAM |
| Gemma 4 12B Q4_K_M | ~7.4 GB |
| Gemma 4 12B BF16 | ~24 GB |
| Android OCR (LiteRT-LM) | ~2.8 GB |

**Rule**: At most one heavy model resident at a time. Release before loading another.

---

## 4. PC Server Optimizations (Prism)

### Current State

- ✅ Whisper-turbo set as default (was small)
- ✅ Lazy loading via ModelManager
- ✅ Idle unload after 10 minutes
- ✅ VAD pre-filter (Silero VAD) to skip silence
- ✅ Auto-split large files (>30s)
- ✅ Gemma4 post-processing with bounded generation

### Recommended Additions

1. **KV cache quantization for Gemma4**: Add `kv_cache_quant="q8_0"` option
2. **Context window sizing**: Dynamic based on input length
3. **Speculative decoding**: Use smaller draft model (Bonsai 8B Q1) for Gemma 12B verification

---

## 5. Android Models for Summarization/Cleanup

### Option A: Gemini Nano (ML Kit GenAI)

**Best for**: Production Android apps with summarization/cleanup/rewrite

| Feature | Details |
|---------|---------|
| Model size | 1.8B-3.25B parameters |
| Quantization | 4-bit |
| Context | 4,096 tokens |
| Latency | Sub-100ms on flagship NPU |
| Device support | Pixel 8+, Samsung S24+, Xiaomi, Motorola |
| Integration | ML Kit GenAI API (`com.google.mlkit:genai-prompt:1.0.0-beta1`) |
| Features | Summarization, Proofreading, Rewriting, Image Description |

**Kotlin Integration:**
```kotlin
implementation("com.google.mlkit:genai-prompt:1.0.0-beta1")

val summarizer = Summarization.getClient()
val summary = summarizer.summarize(text).await()
```

### Option B: Gemma 4 E2B / E4B (LiteRT-LM)

**Best for**: Custom on-device models with full control

| Feature | E2B | E4B |
|---------|-----|-----|
| Parameters | 2.3B effective | 4.5B effective |
| Quantized size | ~3.1 GB (Q4_K_M) | ~5.0 GB (Q4_K_M) |
| Context | 128K tokens | 128K tokens |
| Backend | LiteRT-LM | LiteRT-LM |

### Option C: Bonsai 27B Q1_0 (1-bit)

**Best for**: Extreme compression on devices with limited RAM

| Feature | Details |
|---------|---------|
| Size | ~3.9 GB |
| Quantization | 1-bit (Q1_0) |
| Context | 262K tokens |
| Backend | llama.cpp (requires custom build) |
| Quality | 89.5% of full precision |

### Option D: Qwen 3.5 9B Distill

**Best for**: Reasoning tasks on mobile

| Feature | Details |
|---------|---------|
| Size | ~5.8 GB (Q4_K_M) |
| Parameters | 9B |
| Context | 32K tokens |
| Quantization | Q4_K_M, Q5_K_M, Q6_K, Q8_0 |

---

## 6. Android Implementation Plan

### Phase 1: Gemini Nano (ML Kit GenAI)

Add to `android/app/build.gradle.kts`:
```kotlin
implementation("com.google.mlkit:genai-prompt:1.0.0-beta1")
```

Add summarization endpoint in `ApiServer.kt`:
```kotlin
// POST /v1/text/summarize
// Uses ML Kit Summarization client
```

### Phase 2: Gemma 4 E2B via LiteRT-LM

For custom summarization with quantized weights:
- Download `gemma-4-E2B-it-Q4_K_M.gguf` (~3.1 GB)
- Load via LiteRT-LM CPU backend
- Context: 4,096 tokens (mobile-optimized)

### Phase 3: Bonsai 27B Q1_0 (Optional)

For extreme compression:
- Requires custom llama.cpp build for Android
- ~3.9 GB model size
- 262K context capability

---

## 7. Files Created/Modified

| File | Change |
|------|--------|
| `pc/unified_server.py` | Default whisper model changed from "small" to "turbo" |
| `pc/tests/test_compare_v2.py` | Comparison test: turbo vs large vs nemotron vs gemma4-12B |
| `pc/models/gemma-4-E2B-it/` | DELETED (2B model, replaced by 12B) |
| `pc/models/gemma-4-12B-it-q4/` | Q4_K_M quantized download (in progress) |

---

## 8. Next Steps

1. Wait for Gemma 4 12B Q4_K_M download to complete
2. Run `test_compare_v2.py` to compare all models
3. Add Gemini Nano dependency to Android `build.gradle.kts`
4. Implement summarization endpoint in `ApiServer.kt`
5. Add KV cache quantization option to `gemma_postprocess.py`
6. Test on Android device when reconnected
