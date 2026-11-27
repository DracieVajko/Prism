# Unified contract (PC :8101, Android :8101) — frozen for Odysseus

Base: `http://<host>:8101` where host is `127.0.0.1`, the LAN IP, or the
Tailscale IP (the server binds `0.0.0.0`). No auth: trusted LAN/tailnet only.
All image routes take `model`
(registry id, empty = default) and `request_id` (echoed back).

Legacy: the old `WhisperApi :4444/v1/audio/*` remains as a shim to `:8101/v1/audio/*`
(request/response shape identical). New code always calls `:8101`.

## Endpoints

| Method | Path | In | Out |
|---|---|---|---|
| GET | `/health` | — | `{status, model, current_model, device, available_models, loading, loading_model, busy, queue, audio_model, audio_loaded}` |
| GET | `/v1/models` | — | `{data:[{id...}], currently_loaded_model, loading, loading_model, missing_models, default_image_model, default_audio_model, audio_models}` |
| GET | `/v1/images/progress/{request_id}` | — | `{id, status: running\|done\|error\|loading\|unknown, step, total, percent}` |
| POST | `/v1/images/generations` | JSON `{model, prompt, n 1..4, size WxH, quality, request_id}` | `{created, data:[{b64_json}], request_id}` |
| POST | `/v1/images/edits` | multipart `image + prompt, model, n, size, quality, strength, request_id` | same |
| POST | `/v1/images/inpaint` | JSON `{image b64, mask b64, prompt, strength, model, request_id}` | `{image b64, elapsed, request_id}` |
| POST | `/v1/images/outpaint` | `{image, prompt, margin?|left/right/top/bottom, strength?, steps?}` | `{image, input_size, output_size, margins, engine}` |
| POST | `/v1/images/harmonize` | JSON `{image b64, prompt, strength, model, request_id}` | `{image b64, elapsed, request_id}` |
| POST | `/v1/images/upscale` | JSON `{image b64, scale 2/3/4, model, prompt?, strength?, request_id}` | `{image b64, elapsed, model, scale, engine, request_id}` |
| POST | `/v1/videos/upscale` | multipart `file + scale, model, fps, prompt, strength` | `{request_id, frames, fps, scale, model, engine, elapsed, url, outputs}` |
| POST | `/v1/images/colorize` | JSON `{image b64, model, request_id}` | `{image b64, elapsed, engine, request_id}` | `model` = `deoldify-artistic` (default, most chroma) \| `ddcolor-full` \| `ddcolor-artistic`. Preserves the source resolution and luminance. |
| POST | `/v1/images/restore` | JSON `{image b64, prompt?, strength?, steps?, guidance?, image_guidance?, request_id?}` | `{image b64, elapsed, engine, request_id}` | Z-Image Turbo first (engine `zimage-turbo`), SDXL img2img as fallback with `strength` clamped to <=0.35. 503 `will not load` when neither engine is installed. |
| POST | `/v1/images/caption` | JSON `{image b64, prompt, max_tokens, request_id}` | `{text, elapsed, engine, request_id}` (V2, Qwen3-VL-2B) |
| POST | `/v1/images/ocr` | multipart `file + model, language, task` | `{text, model, language, task}` |
| POST | `/v1/chat/completions` | OpenAI vision subset `{model, messages:[{role, content:str\|[{type:text\|image_url}]}]}` | `{choices:[{message:{content}}]}` (V2, 503 without weights) |
| GET | `/v1/settings/secrets` | - | `{hf_token, llm:{provider,base_url,model,api_key}, _configured}` - secrets masked as `****abcd` |
| PUT | `/v1/settings/secrets` | `{hf_token?, llm?}` | same masked view; masked values are ignored, empty clears |
| POST | `/v1/settings/secrets/test/hf` | - | `{ok, detail}` HuggingFace token check |
| POST | `/v1/settings/secrets/test/llm` | - | `{ok, detail}` provider round-trip |
| GET | `/v1/update/check` | - | `{current_sha, latest_sha, update_available, commits_behind, changelog}` |
| POST | `/v1/update/apply` | - | `{success, message, new_sha?}` (git pull; restart required) |
| GET | `/v1/models/catalog` | — | `{models:[{id, kind, label, size_gb, installed, installed_gb, install_state, install_progress, install_error}]}` |
| POST | `/v1/models/install` | JSON `{id}` | `{id, state}` (download in the background; progress in the catalog). After the download the server runs the model's preparation hook and only then writes `.READY`; states are `idle -> downloading -> preparing -> done` (or `error`). |
| POST | `/v1/models/uninstall` | JSON `{id}` | `{id, removed}` (path guard: only `models/`) |
| GET | `/v1/settings` | — | `{version, port, whisper_url, models_dir, default_image_model, audio_models, engines:{torch,cuda,ffmpeg,ffprobe,ytdlp}}` |
| GET | `/v1/ffmpeg/presets` | — | `{video:{...}, audio:{...}, resize:{...}}` |
| POST | `/v1/ffmpeg/{convert-video\|convert-audio\|extract-audio\|trim\|compress\|info}` | multipart | `{file\|info, filename, request_id, info:{duration,size,format}}` |
| POST | `/v1/audio/transcriptions` | multipart `file|audio + model, device, language` | `{text, language}` or `text/plain` |
| POST | `/v1/audio/translations` | same | `{text}` |
| POST | `/v1/live/sessions` | JSON `{language?, model?, label?}` (defaults `""`, `"turbo"`, `""`) | `{id, sources:["mic","device"], created, language, model, label, closed, descriptors:{mic:<descriptor>, device:<descriptor>}}` |
| POST | `/v1/live/sessions/{sid}/append` | JSON `{source:"mic"\|"device", text, item_id?, elapsed_ms?}` | `{ok, duplicate, bytes, line}`; `400` bad source, `404` unknown session, `409` closed |
| POST | `/v1/live/sessions/{sid}/close` | — | `{ok, closed, mic:<descriptor>, device:<descriptor>, descriptors:{mic,device}, lines:{mic,device}}`; idempotent |
| GET | `/v1/live/sessions/{sid}` | — | `{id, language, model, label, created, sources, descriptors:{mic,device}, lines:{mic,device}, closed, open_sockets, last_activity}` |
| WS | `/v1/realtime` | OpenAI transcription events (see below) | `session.updated`, `input_audio_buffer.committed`, `conversation.item.input_audio_transcription.completed` |
| GET | `/webui` | — | unified HTML (Generate/Edit-Upscale/Transcribe) |

## Batch and video upscale

`POST /v1/batch` accepts upscale jobs with the same fields as the single
endpoint, including Z-Image's `prompt`, `strength` and `seed`:

```json
{"jobs":[
  {"type":"upscale","image":"<b64>","scale":2,"model":"z-image-turbo",
   "prompt":"","strength":0.33},
  {"type":"upscale","image":"<b64>","scale":2,"model":"RealESRGAN_x4v3"}
]}
```

`POST /v1/videos/upscale` extracts frames with ffmpeg, upscales each one, and
remuxes the original audio. Deterministic engines allow up to 900 frames;
Z-Image is capped at **24 frames** because each frame is a diffusion pass
(tens of seconds). Over the cap returns 400 with the frame count and a hint.

## Realtime transcription (WebSocket)

PC and Android expose the same OpenAI transcription-session protocol, used by
Cindy's "Custom realtime ASR -> OpenAI transcription" mode.

Client to server:

- `session.update` — `{"session":{"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{"model":"turbo","language":"en"},"turn_detection":null}}}}`
- `input_audio_buffer.append` — `{"audio": "<base64 PCM16 mono>"}`
- `input_audio_buffer.commit`
- `input_audio_buffer.clear` / `session.finish` (optional)

Server to client:

- `session.updated`
- `input_audio_buffer.committed` — `{"item_id": "item_..."}`
- `conversation.item.input_audio_transcription.completed` — `{"item_id", "content_index":0, "transcript", "usage"}`
- `conversation.item.input_audio_transcription.failed` / `error`

Audio is resampled from the announced rate (24 kHz) to 16 kHz. V1 returns the
final transcript per commit; it does not send partial deltas.

## Live transcription sessions

The four `/v1/live/*` routes above own an in-memory session with **two separate
append-only `.txt` files**, one for `mic` and one for `device`, under `output/text/`
(PC) and `files/output/text/` (Android). Both files are created at open time with
a one-line header and appear in `GET /v1/outputs`. A descriptor has the shape
`{name, kind:"text", size, path:"text/<name>", url:"/v1/outputs/text/<name>"}`.
Appends are flushed immediately as `[mm:ss] <text>` and deduplicated by
`item_id` (a retry returns `{"ok":true,"duplicate":true}` without a second line).
`close` flushes both files; a session idle for 30 minutes with no open socket is
closed automatically.

Captured audio stays in the client's memory and travels through the existing
**`/v1/realtime` WebSocket, whose protocol is unchanged** (Cindy needs no change).
Nothing records audio: no `MediaRecorder` in the browser, no audio file on the
server, and no `.wav`/`.m4a` artifact in `/v1/outputs`.

- **PC localhost-only:** microphone (`getUserMedia`), shared-tab audio
  (`getDisplayMedia`) and `AudioWorklet` need a secure context. The WebUI is served
  over plain HTTP with no TLS, so Live works only at `localhost`/`127.0.0.1`; over
  LAN or Tailscale both source controls are disabled with an explanation. File-based
  transcription still works over LAN. PC tab audio is the **selected browser tab
  only**, not whole-system loopback.
- **Android:** mic uses `AudioRecord`, device audio uses `MediaProjection` playback
  capture with no `VirtualDisplay`. Apps that disallow playback capture yield
  silence — no transcript line is invented. Every `/v1/live/*` call and live socket
  **bypasses Local/Remote/Auto forwarding**, so the session always runs on the
  phone.

## Client compatibility

| Client | Mode | Lenswright setting |
|---|---|---|
| Cindy | Custom realtime ASR -> OpenAI transcription | `ws://<host>:8101/v1/realtime`, model `turbo`, any API key |
| Odysseus | Speech provider `endpoint:<id>` (OpenAI STT) | `http://<host>:8101/v1`, audio model `turbo`/`large` |
| OpenAI SDK / any OpenAI-compatible client | `audio.transcriptions.create` | base URL `http://<host>:8101/v1` |
| Android app | its own server | `ws://<phone>:8101/v1/realtime` + the same HTTP routes |
| Claude Code | built-in dictation streams to Anthropic, no custom ASR URL | not usable |
| Codex CLI | voice is a conversation mode, no custom ASR URL | not usable |

## PC / Android parity

Both products expose the same paths. Android answers `503` with a reason for
features that need the PC: video upscale, PC OCR/VLM models (Android OCR is
`xiaomi-ocr-0`), and optional ASR engines other than whisper.cpp. Model
catalog entries carry `description`, `license`, `size_gb`, `installed` and
`install_state` on both.

## Status codes (identical to ODYSSEUS-CONTRACT + audio)

- `200` + data: success.
- `400`: bad input (missing image, Empty file, bad scale).
- `415`: unsupported audio format. `413`: >100 MB.
- `503 {"error":"loading model","loading_model":..., "retry_after":30}`: cold load — wait, not a failure.
- `503 {"error":"will not load",...}`: does not fit into VRAM or RAM, or canvas >8192px — it fails fast instead of hanging.
- `500`: real server error.

## V1 limits

- Always at most 1 heavy model (SD *or* Whisper). Switching = unload + load.
- Upscale default `RealESRGAN_x4v3`, tile processing, cap 8192px.
- Upscale engines: `swinir-realsr-x4` (deterministic ONNX, 4 GB-safe, x4 then
  resize) and `z-image-turbo` (ESRGAN base + Z-Image img2img refine, prompt and
  strength optional; needs CUDA and `pc/models/z-image-turbo`).
- Audio: 16 kHz mono normalization via ffmpeg (if available), otherwise wav passthrough.
- Mobile: steps clamp 4–8 (thermal throttle after the 3rd generation −40%), Vulkan (not NNAPI).
