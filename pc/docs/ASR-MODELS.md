# ASR models — languages, speed, size, status

Prism's PC server can transcribe with several engines. Pick one per request
with `model=` on `POST /v1/audio/transcriptions` (multipart). `whisper-1`
follows `PRISM_ASR_ENGINE` or defaults to `turbo`.

Sizes are the Hugging Face repository sizes; some apps quote smaller numbers
because they ship ONNX/GGUF/int8 builds. Speeds are the figures published by
the model/app vendors; Prism has not benchmarked every model yet and the real
time depends on GPU/CPU.

| Model id | Languages | Speed (claimed) | Size | Licence | Status |
|---|---|---|---|---|---|
| `turbo` (faster-whisper large-v3-turbo) | 42 | ~2 s | 1.55 GB | MIT | **wired, default** |
| `large` (faster-whisper large-v3) | 42 | slower, best Whisper | 2.9 GB | MIT | wired |
| `nemotron-asr-0.6b` | 40 (streaming) | no wait | 2.55 GB | OpenMDW-1.1 | wired |
| `ivrit-whisper-turbo` | Hebrew | ~2 s | 1.62 GB | Apache-2.0 | wired (CT2) |
| `qwen3-asr-0.6b` | 30 + 22 Chinese dialects | ~2 s | 1.88 GB | Apache-2.0 | wired |
| `qwen3-asr-1.7b` | 30 + 22 Chinese dialects | ~2 s | 4.08 GB | Apache-2.0 | wired, tight on 6 GB |
| `qwencleo-asr-1.7b` | Egyptian/MSA Arabic + English | ~3 s | 4.08 GB | Apache-2.0 | wired |
| `cohere-transcribe` | 14 | <1 s | 4.13 GB | Apache-2.0 | wired |
| `granite-speech-470m` | en, fr, de, es, pt, it, ja | ~1 s | 0.95 GB | Apache-2.0 | wired, tested (20 s endpoint incl. load, 11 s audio) |
| `granite-speech-2b` | en, fr, de, es, pt, it, ja | ~1 s | 5.05 GB | Apache-2.0 | wired, tight on 6 GB |
| `parakeet-v3` | 25 European, auto language | <1 s | 5.73 GB | CC-BY-4.0 | wired, tight on 6 GB |
| `parakeet-v2` | English | <1 s | 2.47 GB | CC-BY-4.0 | wired |
| `parakeet-unified-en` | English (streaming) | no wait | 2.47 GB | see repo card | wired |
| `nb-whisper-medium` | Norwegian (Bokmål/Nynorsk) + English | ~2 s | 3.06 GB | Apache-2.0 | wired |
| `sensevoice-small` | zh, en, yue, ja, ko | <1 s | 0.94 GB | model licence | not wired (FunASR runtime) |
| `gigaam-v3` | Russian | <1 s | 0.45 GB | MIT | not wired (custom runtime) |
| `orukeet` | 25 European | <1 s | 9.85 GB | CC-BY-SA-4.0 | not wired (NeMo, too large) |
| `underdog-bark-3.0` | English | <1 s | 0.39 GB | CC-BY-NC-4.0 | not wired (non-commercial) |

## Which model for which language

- **Slovak / Czech / Polish / most European languages** — `turbo` (Whisper
  large-v3-turbo) or `nemotron-asr-0.6b`; `parakeet-v3` covers 25 European
  languages with automatic language detection, but needs ~5.7 GB.
- **Hebrew** — `ivrit-whisper-turbo` (Whisper fine-tuned by ivrit.ai).
- **Norwegian** — `nb-whisper-medium` (National Library of Norway).
- **Russian** — `turbo`/`large`; `gigaam-v3` is the Russian specialist but its
  runtime is not wired yet.
- **Arabic (Egyptian/MSA, Arabic-English code-switching)** — `qwencleo-asr-1.7b`
  or `qwen3-asr-1.7b`.
- **Chinese, Cantonese, Japanese, Korean** — `qwen3-asr-0.6b`/`1.7b`;
  `sensevoice-small` is the small CJK option once its runtime is added.
- **English only, lowest latency** — `parakeet-v2`, `parakeet-unified-en`,
  `granite-speech-470m`, `cohere-transcribe`.
- **Anything with 42-language coverage** — `turbo` remains the safe default.

## Download and use

```powershell
python pc\download_models.py qwen3-asr-0.6b
python pc\download_models.py granite-speech-470m
python pc\download_models.py ivrit-whisper-turbo
```

```bash
curl -F file=@clip.wav -F model=qwen3-asr-0.6b http://127.0.0.1:8101/v1/audio/transcriptions
```

The models appear in `GET /v1/models/catalog` with kind `audio`, so the WebUI
Models tab can download them.

## Notes

- These are **PC only**. The Android app still uses whisper.cpp with the GGML
  models from its own catalog.
- `parakeet-v3`, `granite-speech-2b`, `qwen3-asr-1.7b` and `qwencleo` sit near
  or above the 6 GB VRAM limit; transformers offloads what does not fit, which
  is slower but works.
- `underdog-bark-3.0` is CC-BY-NC-4.0 (non-commercial), so it is documented but
  not offered for download.
- Granite Speech needs `torchaudio` (installed in the Prism Python env); the
  other transformers ASR models do not.
- CC-BY-4.0 Parakeet models require attribution if you redistribute outputs or
  build a product on them.
