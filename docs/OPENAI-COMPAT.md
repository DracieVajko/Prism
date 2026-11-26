# OpenAI-compatible API (Lenswright PC server)

Lenswright exposes an OpenAI-compatible surface so any tool that speaks the OpenAI
SDK (Python `openai`, curl, LM Studio clients, etc.) can use it out of the box
on `http://127.0.0.1:8101`.

## Endpoints

| Method | Path | OpenAI equivalent | Notes |
|---|---|---|---|
| GET | `/v1/models` | List models | Returns image/audio/VLM/OCR models with `id`, `object`, `owned_by`. |
| POST | `/v1/audio/transcriptions` | Audio transcription | `file` (multipart), `model`, `language`, `response_format` (json/text/srt/vtt). Accepts audio + video containers. Extra Lenswright options: `output_format`, `post_process`, `use_vad`, `max_chars_per_chunk`. |
| POST | `/v1/audio/translations` | Audio translation | Same as above, translates to English. |
| POST | `/v1/images/generations` | Image generation | `prompt`, `size`, `n`, `model`. |
| POST | `/v1/images/edits` | Image edit | `image`, `prompt`, `mask`. |
| POST | `/v1/images/inpaint` | — | Lenswright-specific inpainting. |
| POST | `/v1/images/upscale` | — | Lenswright-specific upscale. |
| POST | `/v1/chat/completions` | Chat completions | **Vision-only**: requires an image in the message content (`image_url` data:base64). Returns OpenAI-shaped `choices[0].message.content`. Text-only chat requires an external LLM provider (LM Studio / Ollama) configured in Lenswright. |
| POST | `/v1/ffmpeg/{op}` | — | Media tools (info, convert-*, extract-audio, trim, compress, run). |
| GET | `/v1/outputs` | — | List produced artifacts. |

## Example: transcription (curl)

```
curl -X POST http://127.0.0.1:8101/v1/audio/transcriptions \
  -F "file=@recording.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

## Example: transcription (Python openai SDK)

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8101/v1", api_key="not-needed")
with open("recording.mp3", "rb") as f:
    r = client.audio.transcriptions.create(model="whisper-1", file=f)
print(r.text)
```

## Example: image generation (Python)

```python
r = client.images.generate(prompt="a cozy cabin at sunset", size="512x512", n=1)
print(r.data[0].url)   # or b64_json depending on response
```

## Notes
- No API key required (local server). Set `api_key="not-needed"` / any string.
- Authentication is none by default — bind to 127.0.0.1. Use `-Remote` only on a trusted LAN.
- `model` names must match `/v1/models` (e.g. `whisper-1`, image model ids). Unknown names fall back to the default.