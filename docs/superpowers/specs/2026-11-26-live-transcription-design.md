# Live transcription to text files (design)

Date: 2026-11-26
Status: draft — awaiting approval
Scope: PC server + WebUI (`pc/`), Android app (`android/`). Shared API on both.

## Intent

Add a live transcription session that captures audio continuously, transcribes
it while the session runs, and writes two plain-text files — one for microphone
audio and one for screen/device audio. **No audio recording is ever saved.**
The transcript files are the only artifacts.

This serves a user who wants a running transcript of a meeting or a screen
session without accumulating large audio files or doing a separate upload step.
It is the implementable subset of `docs/MEETING-TRANSCRIBE-PLAN.md` (dual-channel
capture + live transcript), intentionally dropping video recording and LLM
meeting intelligence for now.

## Goals

- Start a session, see transcript lines appear in two separate `.txt` files
  while it runs, stop it, and find both files in the normal Outputs list.
- Capture microphone on both platforms. Capture screen/device audio on both:
  shared-tab audio in the PC browser, `MediaProjection` audio on Android.
- Reuse the existing `/v1/realtime` WebSocket for speech-to-text; do not change
  its protocol (Cindy depends on it).
- Never write captured audio to disk — only in-memory PCM buffers.
- Work fully offline on Android in Local mode, and on the PC without extra
  services beyond the existing Whisper engine.

## Non-goals

- Whole-system audio loopback on the PC. A web page cannot capture all system
  audio; only a shared browser tab's audio is available through
  `getDisplayMedia`. Document this limit in the UI and the contract.
- PC live capture over LAN/Tailscale HTTP. Browser audio capture needs a secure
  context; live capture is localhost-only by decision (no TLS in scope).
- Screen **video** recording, screenshots, or replay.
- True streaming/partial ASR deltas. The current socket returns a final
  transcript per `commit`; this design commits on a short cadence instead.
- Diarisation, speaker labels beyond the source label, punctuation SPA, or LLM
  meeting summaries (the existing post-processing remains available for file
  transcription only).
- Changing the file-based Transcribe flow, `/v1/audio/transcriptions`, or the
  Cindy realtime contract.

## Constraints

- One heavy model resident at a time; a live session keeps the audio engine
  loaded and must hold the resident gate for the duration of a commit.
- PC: FastAPI/Uvicorn, `_store_output` writes under `pc/output/<kind>/`,
  `/v1/outputs` scans that tree. Audio gate is `ResidentModelGate`.
- Android: Ktor/Netty, `OutputStore` writes under app-external `files/output/`
  and mirrors to `Download/Lenswright`. `FOREGROUND_SERVICE_DATA_SYNC` exists;
  microphone capture needs additional permissions on Android 14+.
- `RECORD_AUDIO` is already declared; `MediaProjection` and
  `FOREGROUND_SERVICE_MICROPHONE` are not.
- `MediaProjection` internal-audio capture only works for apps that allow
  playback capture (`android:allowAudioPlaybackCapture`); many do not. Surface
  this honestly rather than promising system audio everywhere.
- English-only UI strings, comments, docs and log messages.

## Approaches considered

- **A. Small live-session API; server writes files (chosen).** Three endpoints
  create/append/close a session. The server appends each transcript fragment
  immediately to two files. Clients only capture and post text. Cindy's socket
  is untouched; files are visible in `/v1/outputs` during the session.
- **B. Extend `/v1/realtime` with session/source fields.** One transport, but it
  changes a frozen protocol that an external client uses; every addition must
  be optional and ignored by Cindy. Higher regression risk for little gain.
- **C. Client-only accumulation.** No server changes; each client writes the
  two files at Stop. Simplest, but files do not exist during the session, are
  invisible to `/v1/outputs`, and a crash loses the transcript. Rejected.

## API design (identical on PC and Android)

### `POST /v1/live/sessions`

Request: `{"language": "sk", "model": "turbo", "label": "meeting"}` — all
optional. Response:

```json
{
  "id": "live_20261126-131500-a1b2c3",
  "mic":       {"name": "20261126-131500-live_...-mic.txt", "path": "text/...", "url": "/v1/outputs/text/..."},
  "device":    {"name": "20261126-131500-live_...-device.txt", "path": "text/...", "url": "/v1/outputs/text/..."},
  "created": 1795...,
  "sources": ["mic", "device"]
}
```

Both files are created at open time with a short header line
(`# Live transcript — source: mic — language: sk`) so they are present and
discoverable immediately, even before the first utterance. Unknown source
names are rejected by the append endpoint, not by open.

### `POST /v1/live/sessions/{id}/append`

Request: `{"source": "mic"|"device", "text": "...", "item_id": "item_...",
"elapsed_ms": 5120}`. Response: `{"ok": true, "bytes": 42, "line": 3}`.

Rules:
- `source` must be one of the two opened sources; otherwise 400.
- Empty/whitespace `text` is accepted and ignored (no blank line written).
- A fragment is appended as one line: `[00:05] <text>` using `elapsed_ms`.
- Files are opened in append mode, flushed on every append, so a reader sees
  new lines without waiting for session close.
- The session must exist and be open; otherwise 404 / 409 if closed.
- Append is **idempotent on `item_id`**: the server keeps a per-session set of
  seen ids and returns `{"ok": true, "duplicate": true}` without writing a
  second line when an id repeats. This makes a client retry after a dropped
  HTTP response safe.

### `POST /v1/live/sessions/{id}/close`

Response: `{"ok": true, "mic": <descriptor>, "device": <descriptor>,
"lines": {"mic": 12, "device": 9}}`. Flushes and marks the session closed.
Closing twice is idempotent. Abandoned sessions are closed by a 30-minute idle
sweep so a crashed client cannot leak a file handle. For the sweep, **idle means
no open WebSocket for that session AND 30 minutes since the last append or
last socket activity** — a quiet meeting with open sockets is never swept.

### `GET /v1/live/sessions/{id}` (optional, small)

Status, per-source line counts and descriptors; useful for the UI and tests.

Path/naming follows the existing output rules: `output/text/` (PC) and
`files/output/text/` (Android), filename
`YYYYmmdd-HHMMSS-<sanitized session id>-<source>.txt` with `-n` collision
suffix. Session ids are generated server-side and sanitised before use; no
client-supplied path component reaches the filesystem.

## Client flow (both platforms)

For each enabled source, the client:

1. obtains a PCM16 16 kHz mono stream (see below);
2. opens one WebSocket to `/v1/realtime` with `session.update`
   (`rate: 16000`, `model`, `language`);
3. accumulates ~5 s of PCM, sends `input_audio_buffer.append` (base64) then
   `input_audio_buffer.commit`;
4. on `conversation.item.input_audio_transcription.completed`, POSTs
   `{source, text, item_id, elapsed_ms}` to the append endpoint;
5. repeats until Stop, then sends `session.finish`, closes the socket, and
   calls the close endpoint.

Two sources mean two sockets and two append streams; source identity is kept
client-side and never mixed into one buffer. A stream that fails to start is
reported as a per-source error and the other source continues.

Commit cadence of 5 s is a design call: it keeps perceived latency low while
giving Whisper enough audio for a stable final transcript. It is configurable
in code (`LIVE_COMMIT_MS`) and may be tuned during implementation.

**Back-pressure policy.** Audio capture never pauses. Commits are queued with
at most two in flight per source; if the queue is full the oldest unprocessed
chunk is dropped and a warning is logged (never silently). On Stop, the client
stops capture, then **drains in-flight commits and waits for their
`completed` events before calling close** (with a bounded timeout, after which
the session is closed and the loss reported in the UI). This guarantees the
last utterance is normally in the file.

## PC WebUI decisions

- Add a **Live** block to the Transcribe tab: two source checkboxes
  (**Microphone**, **Shared tab audio**), language/model selects, Start/Stop,
  a tail preview of the last line per source, and links to both files.
- Microphone: `navigator.mediaDevices.getUserMedia({audio:true})`.
- Shared tab audio: `getDisplayMedia({video:true, audio:true})`; the video
  track is stopped immediately and only the audio track is kept. If no audio
  track is returned, show a clear "no tab audio shared" error for that source.
- PCM capture: `AudioContext` + `AudioWorklet` (fallback
  `ScriptProcessorNode`) → downsample to 16 kHz → float32 to int16 → base64.
- Streams and the AudioContext are stopped/closed on Stop; no `MediaRecorder`
  is created, so no audio file exists in the browser or on the server.
- On load, check `window.isSecureContext`. When false (LAN/Tailscale HTTP),
  disable both Live source controls and show the localhost-only explanation;
  never let the user start a session that cannot capture audio.

## Android decisions

- Add a **Live** section to the Transcribe tab: two switches
  (**Microphone**, **Device audio**), language/model, Start/Stop, a tail
  preview, and an "Open outputs" link.
- Microphone: `AudioRecord` with `AudioSource.MIC`, 16 kHz mono PCM16, read on
  a background thread owned by `DiffusionService`.
- Device audio: `MediaProjection` obtained via
  `MediaProjectionManager.createScreenCaptureIntent()`, then
  `AudioPlaybackCaptureConfiguration` → a second `AudioRecord`. Only audio is
  captured; no `VirtualDisplay` is created. Apps that disallow playback
  capture simply yield silence — the UI says so up front.
- Permissions/manifest: keep `RECORD_AUDIO`; add
  `FOREGROUND_SERVICE_MICROPHONE` and
  `FOREGROUND_SERVICE_MEDIA_PROJECTION`, extend `foregroundServiceType` to
  include `microphone|mediaProjection` while a live session is active. Runtime
  permission requests use the existing request-code pattern in `MainActivity`.
- Live capture always talks to the **local** server at `127.0.0.1:8101`.
  Local/Remote/Auto forwarding is bypassed for **all** `/v1/live/*` calls and
  for live sockets, not just sockets; otherwise a session opened on the phone
  would receive appends on a PC with no such session. Remote mode is suspended
  for the duration of a live session and restored at Stop. Document this.
- The session (state, both WebSockets, the append loop) is owned by
  `DiffusionService`, not the Activity, so it survives backgrounding and
  screen-off. The UI binds to the service and mirrors status/tail lines.
- Pressing Live **auto-starts the local API server** through the same path as
  the START button, with the same permission/error handling. A fresh app does
  not require a separate manual start.
- The canonical files are written by `OutputStore` under `files/output/text/`;
  the final content is mirrored to `Download/Lenswright` once at close,
  matching existing output behavior without rewriting the file on every append.

## Error and lifecycle behavior

- No engine installed → the socket returns `...failed`; the client shows a
  per-source error and the live session still opens (with an empty file) so
  the user can retry after downloading the model.
- Permission denied → only that source is disabled; the other continues.
- Server restart during a session → the next append returns 404; the client
  stops and reports that the session was lost. The files written so far remain.
- 30-minute idle sweep closes abandoned sessions and flushes their files.
- All responses carry the same error shape used elsewhere
  (`{"error": "...", "detail": "..."}`); no invented success states.

## Tests

- PC: unit tests for session open/append/close, separate files, header/line
  format, empty-text handling, path guard, idempotent close, 404 after restart
  of state, and the idle sweep. A WebSocket integration test with a stubbed
  transcriber drives two sources through commit → append → two files and
  asserts no audio file is created anywhere under `output/`.
- Android: instrumented HTTP test for the same session endpoints and
  `OutputStore` file placement; a permission-gated manual device check for
  `AudioRecord` and `MediaProjection` capture (automation cannot grant the
  projection consent dialog).
- Update `docs/CHECKLIST.md`, `docs/ARCHITECTURE.md`, `pc/docs/UNIFIED-CONTRACT.md`
  (new endpoints + the browser tab-audio limit) and the lenswright SKILL.md.

## Decision: PC live capture is localhost-only

`getUserMedia`, `getDisplayMedia` and `AudioWorklet` require a secure context.
The WebUI stays on plain HTTP; **live capture is supported only when the UI is
opened on `localhost` (or `127.0.0.1`)**. Over a LAN or Tailscale address the
Live source controls are disabled with an explicit message explaining that
browser audio capture needs localhost. No TLS work is in scope. This is the
user's choice; file-based transcription remains available over LAN as today.

## Builder-check decisions (settled here, not left open)

- Android forwarding: bypassed for the whole live API and live sockets.
- Android session ownership: `DiffusionService`, UI mirrors state.
- Android server: Live auto-starts it.
- Back-pressure: capture never pauses; max two commits in flight; oldest
  unprocessed chunk logged and dropped when full; Stop drains in-flight
  commits before close.
- Idle sweep: no open sockets **and** 30 minutes without append/socket traffic.
- Append: idempotent on `item_id`.

## What is left to the builder

- Exact `AudioWorklet` processor code and the downsample filter quality.
- `GET /v1/live/sessions/{id}` ships in v1 (a few lines; makes UI and tests
  easier).
- Retention of closed-session metadata after the sweep.
- UI polish (tail preview length, button states) — choose what fits the
  existing WebUI/Android styling.
