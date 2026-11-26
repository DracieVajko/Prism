# Live transcription to text files Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live transcription sessions on PC and Android that write separate microphone and device-audio `.txt` files while capturing, without ever saving audio.

**Architecture:** A small `/v1/live/sessions` API owns two append-only transcript files per session. Clients capture PCM16 in memory, send it through the existing `/v1/realtime` WebSocket, and POST each returned transcript to the append endpoint. No audio bytes are written anywhere. Cindy's realtime protocol is unchanged.

**Tech Stack:** Python 3/FastAPI (PC), Kotlin/Ktor + OkHttp + `AudioRecord`/`MediaProjection` (Android), plain JavaScript `AudioContext` (WebUI), pytest + httpx tests.

**Spec:** `docs/superpowers/specs/2026-11-26-live-transcription-design.md`

## Global Constraints

- Sources are exactly `"mic"` and `"device"`; two separate files per session.
- Commit cadence `LIVE_COMMIT_MS = 5000`; at most two commits in flight per source; capture never pauses; oldest unprocessed chunk is dropped with a logged warning when the queue is full.
- Stop drains in-flight commits and waits for their `completed` events before close (bounded timeout; report loss in UI).
- Append is idempotent on `item_id` (duplicate → no second line).
- Idle sweep: 1800 s, only when no socket is open for the session and no append/socket traffic occurred.
- File header: `# Live transcript - source: <source> - language: <language>`; appended line: `[<mm:ss>] <text>`. Empty text writes nothing.
- File location: `pc/output/text/` (PC), `files/output/text/` (Android). Filename `YYYYmmdd-HHMMSS-<safe session id>-<source>.txt`.
- PC live capture is localhost-only; no TLS; WebUI disables Live controls when `window.isSecureContext` is false.
- Android live API and sockets bypass Local/Remote/Auto forwarding; Live auto-starts the local server; `DiffusionService` owns capture.
- No audio file may be created on either platform; English-only text everywhere.

## Review Focus

- A LAN/Tailscale HTTP WebUI has no secure context: Live controls must be visibly disabled with an explanation, never a button that starts and silently fails. Test: `pc/tests/test_webui_live.py::test_secure_context_guard`.
- An app that disallows playback capture yields silence on the device source: no transcript line must be invented. Test: `pc/tests/test_live_session.py::test_empty_text_writes_nothing` and manual device checklist in Task 6.
- A quiet meeting (30 min silence, sockets open) must not be swept. Test: `pc/tests/test_live_session.py::test_sweep_skips_open_sockets`.
- A retry after a dropped HTTP response must not duplicate a line. Test: `pc/tests/test_live_session.py::test_duplicate_item_id_is_ignored`.
- Stop with a commit in flight must keep the last utterance. Test: `android/tests/test_android_live_sessions.py::test_pending_commit_drained_before_close` (controller-level, fake socket).

---

### Task 1: PC live session store

**Files:**
- Create: `pc/unified/live_session.py`
- Test: `pc/tests/test_live_session.py`

**Interfaces:**
- Consumes: a `Path` output directory passed to `LiveSessionStore.__init__`.
- Produces: `SOURCES = ("mic", "device")`; exceptions `UnknownSession`, `BadSource`, `SessionClosed`; `LiveSessionStore.create(language, model, label) -> dict`, `append(session_id, source, text, item_id, elapsed_ms) -> dict`, `close(session_id) -> dict`, `status(session_id) -> dict`, `sweep_idle(max_idle_s=1800) -> int`, `mark_socket(session_id, delta) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# pc/tests/test_live_session.py
import pytest
from pathlib import Path
from unified.live_session import (
    SOURCES, LiveSessionStore, UnknownSession, BadSource, SessionClosed,
)

@pytest.fixture
def store(tmp_path):
    return LiveSessionStore(tmp_path)

def test_open_creates_two_files(store):
    s = store.create(language="sk", model="turbo", label="meeting")
    assert set(s["sources"]) == set(SOURCES)
    for source in SOURCES:
        p = Path(store.status(s["id"])["descriptors"][source]["path"])
        assert p.exists()
        assert p.read_text(encoding="utf-8").startswith(
            "# Live transcript - source: " + source + " - language: sk")
        assert p.parent.name == "text"

def test_append_writes_separate_timestamped_lines(store):
    s = store.create(language="en")
    store.append(s["id"], "mic", "hello", "item_1", 5000)
    store.append(s["id"], "device", "world", "item_2", 12000)
    mic = Path(store.status(s["id"])["descriptors"]["mic"]["path"]).read_text(encoding="utf-8")
    dev = Path(store.status(s["id"])["descriptors"]["device"]["path"]).read_text(encoding="utf-8")
    assert "[00:05] hello" in mic and "world" not in mic
    assert "[00:12] world" in dev and "hello" not in dev

def test_empty_text_writes_nothing(store):
    s = store.create()
    r = store.append(s["id"], "mic", "   ", "item_empty", 0)
    assert r["bytes"] == 0
    assert len(Path(store.status(s["id"])["descriptors"]["mic"]["path"]).read_text(encoding="utf-8").splitlines()) == 1

def test_duplicate_item_id_is_ignored(store):
    s = store.create()
    store.append(s["id"], "mic", "hello", "item_1", 1000)
    r = store.append(s["id"], "mic", "hello again", "item_1", 2000)
    assert r["duplicate"] is True
    text = Path(store.status(s["id"])["descriptors"]["mic"]["path"]).read_text(encoding="utf-8")
    assert "hello again" not in text

def test_bad_source_and_unknown_session(store):
    s = store.create()
    with pytest.raises(BadSource):
        store.append(s["id"], "screen", "x", "i", 0)
    with pytest.raises(UnknownSession):
        store.append("missing", "mic", "x", "i", 0)

def test_close_is_idempotent_and_then_closed(store):
    s = store.create()
    assert store.close(s["id"])["ok"] is True
    assert store.close(s["id"])["ok"] is True
    with pytest.raises(SessionClosed):
        store.append(s["id"], "mic", "x", "i", 0)

def test_sweep_skips_open_sockets(store):
    s = store.create()
    store.mark_socket(s["id"], 1)
    store.last_activity(s["id"], 0)  # simulate 30 minutes of silence
    assert store.sweep_idle(max_idle_s=1800) == 0
    store.mark_socket(s["id"], -1)
    assert store.sweep_idle(max_idle_s=1800) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd pc && python -m pytest tests/test_live_session.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'unified.live_session'`.

- [ ] **Step 3: Implement `pc/unified/live_session.py`**

Define `SOURCES`, the three exceptions, a `dataclass LiveSession` holding `id, language, model, label, created, descriptors, lines, seen, closed, last_activity, open_sockets, paths`, and `LiveSessionStore` with the signatures in Interfaces. Generate ids as `live_<YYYYmmdd-HHMMSS>_<6 hex>`; sanitise with the same character filter as `_store_output` (`isalnum() or in "-_"`). Write headers on create, append one `[mm:ss] text` line per non-empty fragment, flush every append, dedupe on `item_id`, update `last_activity` on append and socket changes, and close (flush) files on `close`/`sweep_idle`. Return descriptors with `name/kind/size/path/url` matching `_store_output`.

- [ ] **Step 4: Add `last_activity(session_id, ts: float) -> None` to the store**

This helper exists only to make the idle test deterministic; keep it public and document it as test/support use.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd pc && python -m pytest tests/test_live_session.py -q`
Expected: `7 passed`.

- [ ] **Step 6: Commit**

```bash
git add pc/unified/live_session.py pc/tests/test_live_session.py
git commit -m "feat: PC live transcription session store with separate mic/device files"
```

---

### Task 2: PC live session API and idle sweep

**Files:**
- Modify: `pc/unified/media_routes.py` (add `get_live_store()` and four routes near the `/v1/outputs` block at ~2500)
- Modify: `pc/unified_server.py` (idle loop at ~107-134)
- Test: `pc/tests/test_live_session_api.py`

**Interfaces:**
- Consumes: `LiveSessionStore`, exceptions from Task 1; `router`, `_us()`, `JSONResponse`.
- Produces: `get_live_store() -> LiveSessionStore`; routes `POST /v1/live/sessions`, `POST /v1/live/sessions/{sid}/append`, `POST /v1/live/sessions/{sid}/close`, `GET /v1/live/sessions/{sid}`.

- [ ] **Step 1: Write the failing API tests**

```python
# pc/tests/test_live_session_api.py
from fastapi.testclient import TestClient
import unified_server as us

def test_live_api_open_append_close(tmp_path, monkeypatch):
    monkeypatch.setattr(us, "OUTPUT_DIR", tmp_path)
    import unified.media_routes as mr
    mr._LIVE_STORE = None
    client = TestClient(us.app)
    s = client.post("/v1/live/sessions", json={"language": "sk"}).json()
    assert set(s["sources"]) == {"mic", "device"}
    a = client.post(f"/v1/live/sessions/{s['id']}/append",
                    json={"source": "mic", "text": "hello", "item_id": "i1", "elapsed_ms": 1000})
    assert a.status_code == 200 and a.json()["bytes"] > 0
    assert client.post(f"/v1/live/sessions/{s['id']}/append",
                       json={"source": "screen", "text": "x"}).status_code == 400
    assert client.get(f"/v1/live/sessions/{s['id']}").json()["lines"]["mic"] == 2  # header + text
    assert client.post(f"/v1/live/sessions/{s['id']}/close").json()["ok"] is True

def test_live_session_files_are_visible_in_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(us, "OUTPUT_DIR", tmp_path)
    client = TestClient(us.app)
    s = client.post("/v1/live/sessions", json={}).json()
    names = [f["name"] for f in client.get("/v1/outputs").json()["files"]]
    assert any(n.endswith("-mic.txt") for n in names)
    assert any(n.endswith("-device.txt") for n in names)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd pc && python -m pytest tests/test_live_session_api.py -q`
Expected: FAIL with 404 on `/v1/live/sessions`.

- [ ] **Step 3: Add the store accessor and routes in `pc/unified/media_routes.py`**

```python
_LIVE_STORE = None
def get_live_store():
    global _LIVE_STORE
    if _LIVE_STORE is None:
        from unified.live_session import LiveSessionStore
        _LIVE_STORE = LiveSessionStore(_us().OUTPUT_DIR)
    return _LIVE_STORE
```

Add Pydantic models `LiveSessionCreate(language="", model="turbo", label="")`, `LiveAppend(source, text="", item_id="", elapsed_ms=0)`. Map `UnknownSession → 404`, `BadSource → 400`, `SessionClosed → 409` with `{"error": "...", "detail": "..."}`. The `GET` route returns `status()` or 404.

- [ ] **Step 4: Call the idle sweep from `pc/unified_server.py`**

Inside the existing 30 s `_idle_unloader` loop, add a guarded local import and `get_live_store().sweep_idle()` in a `try/except Exception` so a sweep failure cannot stop the unloader.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd pc && python -m pytest tests/test_live_session_api.py tests/test_live_session.py -q`
Expected: `9 passed`.

- [ ] **Step 6: Commit**

```bash
git add pc/unified/media_routes.py pc/unified_server.py pc/tests/test_live_session_api.py
git commit -m "feat: PC live session API with separate transcript files and idle sweep"
```

---

### Task 3: PC WebUI live capture

**Files:**
- Modify: `pc/webui_unified.html` (Transcribe tab form at ~184-204; JS near `generateTr` at ~789)
- Test: `pc/tests/test_webui_live.py`

**Interfaces:**
- Consumes: `/v1/live/sessions` API from Task 2; `/v1/realtime` WebSocket (unchanged).
- Produces: DOM ids `liveMic`, `liveDevice`, `btnLiveStart`, `btnLiveStop`, `liveStatus`, `liveTailMic`, `liveTailDevice`, `liveLinkMic`, `liveLinkDevice`; JS functions `startLive()`, `stopLive()`, `appendLive(sid, source, text, itemId, elapsedMs)`.

- [ ] **Step 1: Write the failing static test**

```python
# pc/tests/test_webui_live.py
from pathlib import Path

HTML = Path(__file__).resolve().parents[1] / "webui_unified.html"

def test_live_controls_present_and_secure_context_guard():
    s = HTML.read_text(encoding="utf-8")
    for ident in ("liveMic", "liveDevice", "btnLiveStart", "btnLiveStop",
                  "liveStatus", "liveTailMic", "liveTailDevice",
                  "liveLinkMic", "liveLinkDevice"):
        assert f'id="{ident}"' in s, ident
    assert "isSecureContext" in s
    assert "getUserMedia" in s and "getDisplayMedia" in s
    for fn in ("startLive", "stopLive", "appendLive"):
        assert f"function {fn}" in s, fn

def test_live_never_uses_media_recorder():
    s = HTML.read_text(encoding="utf-8")
    assert "MediaRecorder" not in s
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd pc && python -m pytest tests/test_webui_live.py -q`
Expected: FAIL on missing `id="liveMic"`.

- [ ] **Step 3: Add the Live block to the Transcribe tab**

Insert after the existing Transcribe controls, before `txtout`: two checkboxes (`liveMic`, `liveDevice`), a language select `liveLang` mirroring `trlang`, `btnLiveStart`/`btnLiveStop`, `liveStatus`, two tail previews and two file links. `btnLiveStop` starts disabled.

- [ ] **Step 4: Implement the capture JavaScript**

`startLive()` checks `window.isSecureContext`; if false, sets `liveStatus` to "Live capture needs localhost" and returns. Otherwise POST `/v1/live/sessions`, then for each checked source open a stream (`getUserMedia({audio:true})` / `getDisplayMedia({video:true,audio:true})`, stopping the video track immediately), build an `AudioContext` with `AudioWorklet` (fallback `ScriptProcessorNode`), downsample to 16 kHz int16, buffer 5000 ms, and send `append`/`commit` on one WebSocket per source. On `...transcription.completed`, call `appendLive()` and update the tail. `stopLive()` sends `session.finish`, waits for pending completions (bounded), POSTs close, stops streams/contexts, and enables the links. No `MediaRecorder` anywhere.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd pc && python -m pytest tests/test_webui_live.py -q`
Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add pc/webui_unified.html pc/tests/test_webui_live.py
git commit -m "feat: WebUI live transcription to separate mic/device text files"
```

---

### Task 4: Android live session store and API

**Files:**
- Create: `android/app/src/main/java/eu/dracievajko/lenswright/LiveSessionStore.kt`
- Modify: `android/app/src/main/java/eu/dracievajko/lenswright/OutputStore.kt` (add `liveFile`, `publish`)
- Modify: `android/app/src/main/java/eu/dracievajko/lenswright/ApiServer.kt` (routing ~533; interceptor ~525)
- Test: `android/tests/test_android_live_sessions.py`

**Interfaces:**
- Consumes: `OutputStore.root/kindDir/publish`; Ktor `call.receiveText()`, `Gson`/`JSONObject`.
- Produces: `LiveSessionStore.create/append/close/status/sweepIdle/markSocket/lastActivity`; routes `POST /v1/live/sessions`, `POST /v1/live/sessions/{sid}/append`, `POST /v1/live/sessions/{sid}/close`, `GET /v1/live/sessions/{sid}`.

- [ ] **Step 1: Write the failing device test**

```python
# android/tests/test_android_live_sessions.py
import httpx
import pytest
BASE = "http://127.0.0.1:8199"
c = httpx.Client(base_url=BASE, timeout=120.0)

def test_live_session_api_and_separate_files():
    s = c.post("/v1/live/sessions", json={"language": "sk"}).json()
    assert set(s["sources"]) == {"mic", "device"}
    sid = s["id"]
    assert c.post(f"/v1/live/sessions/{sid}/append",
                  json={"source": "mic", "text": "hello", "item_id": "i1", "elapsed_ms": 1000}).status_code == 200
    assert c.post(f"/v1/live/sessions/{sid}/append",
                  json={"source": "screen", "text": "x"}).status_code == 400
    st = c.get(f"/v1/live/sessions/{sid}").json()
    assert st["lines"]["mic"] == 2 and st["lines"]["device"] == 1
    assert c.post(f"/v1/live/sessions/{sid}/append",
                  json={"source": "mic", "text": "again", "item_id": "i1"}).json()["duplicate"] is True
    assert c.post(f"/v1/live/sessions/{sid}/close").json()["ok"] is True
    names = [f["name"] for f in c.get("/v1/outputs").json()["files"]]
    assert any(n.endswith("-mic.txt") for n in names)
    assert not any(n.endswith(".m4a") or n.endswith(".wav")
                   for n in names if "-mic" in n or "-device" in n)

@pytest.mark.skip(reason="implemented with the controller in Task 5")
def test_pending_commit_drained_before_close():
    # Controller-level: with a fake socket holding one in-flight commit,
    # stop() must wait for its completed event before close() is called.
    # Implement against LiveCapture.awaitPending(timeoutMs) in Task 5.
    pass
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd android/tests && python test_android_live_sessions.py`
Expected: FAIL with 404 on `/v1/live/sessions` (requires device + `adb forward tcp:8199 tcp:8101`).

- [ ] **Step 3: Add `liveFile` and `publish` to `OutputStore.kt`**

`fun liveFile(ctx: Context, sessionId: String, source: String): File` → `output/text/YYYYmmdd-HHmmss-<safe sessionId>-<source>.txt` (reuse the existing `-n` collision rule). `fun publish(ctx: Context, file: File)` → copy one file to `Download/Lenswright` using the existing MediaStore block in `store()`. Do not change `store()`.

- [ ] **Step 4: Implement `LiveSessionStore.kt`**

Mirror the PC store: `ConcurrentHashMap` of sessions, header on create, append with `[mm:ss]` lines, flush, `item_id` dedupe, `sweepIdle`, `markSocket`, `lastActivity`. Exceptions: `NoSuchElementException` (unknown), `IllegalArgumentException` (bad source), `IllegalStateException` (closed).

- [ ] **Step 5: Add the four routes in `ApiServer.kt`**

Parse JSON with `JSONObject`, map exceptions to 404/400/409 with `{"error","detail"}`. Add `call.request.path().startsWith("/v1/live/")` to the forwarding-interceptor bypass condition so live sessions are always local.

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd android/tests && python test_android_live_sessions.py`
Expected: first test passes; second remains skipped until Task 5.

- [ ] **Step 7: Commit**

```bash
git add android/app/src/main/java/eu/dracievajko/lenswright/LiveSessionStore.kt \
  android/app/src/main/java/eu/dracievajko/lenswright/OutputStore.kt \
  android/app/src/main/java/eu/dracievajko/lenswright/ApiServer.kt \
  android/tests/test_android_live_sessions.py
git commit -m "feat: Android live session API with separate transcript files"
```

---

### Task 5: Android capture controller and permissions

**Files:**
- Create: `android/app/src/main/java/eu/dracievajko/lenswright/LiveCapture.kt`
- Modify: `android/app/build.gradle.kts` (add `ktor-client-websockets:$ktor` beside line 152)
- Modify: `android/app/src/main/AndroidManifest.xml`
- Modify: `android/app/src/main/java/eu/dracievajko/lenswright/DiffusionService.kt`
- Test: `android/tests/test_android_live_sessions.py` (complete the controller test)

**Interfaces:**
- Consumes: `LiveSessionStore`, `HttpClient`, `AudioRecord`, `MediaProjection`.
- Produces: `object LiveCapture` with `start(ctx, sessionId, sources, model, language, projectionIntent) -> Unit`, `stop() -> Unit`, `status() -> Map<String, Any>`, `awaitPending(timeoutMs: Long) -> Boolean`, `attach(ctx)`.

- [ ] **Step 1: Implement the failing controller test**

Replace the `pass` in `test_pending_commit_drained_before_close` with a JVM unit test (place it in `android/app/src/test/java/eu/dracievajko/lenswright/LiveCaptureTest.kt`) that injects a fake commit queue and asserts `awaitPending(2000)` waits for a `completed` callback before returning true, and returns false on timeout. Add `fun setPendingForTest(count: Int)` and `fun completePendingForTest()` hooks on `LiveCapture`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd android && .\gradlew.bat :app:testDebugUnitTest --tests '*LiveCaptureTest*'`
Expected: FAIL — `LiveCapture` unresolved.

- [ ] **Step 3: Add permissions and client dependency**

In `AndroidManifest.xml` keep `RECORD_AUDIO`, add `FOREGROUND_SERVICE_MICROPHONE` and `FOREGROUND_SERVICE_MEDIA_PROJECTION`, and set `android:foregroundServiceType="dataSync|microphone|mediaProjection"` on the service. In `build.gradle.kts` add `implementation("io.ktor:ktor-client-websockets:$ktor")`.

- [ ] **Step 4: Implement `LiveCapture.kt`**

Use two `AudioRecord` instances at 16 kHz mono PCM16 (mic `AudioSource.MIC`; device via `MediaProjection.createAudioRecord(AudioPlaybackCaptureConfiguration.Builder(projection).addMatchAllExcludedActivities... build(), format)`), read on `Dispatchers.IO`, buffer 5000 ms, base64-encode and send `input_audio_buffer.append` + `commit` on one OkHttp/Ktor WebSocket per source to `ws://127.0.0.1:8101/v1/realtime`. On `...transcription.completed`, POST to `/v1/live/sessions/{id}/append` with `source/item_id/elapsed_ms`. Bound queue to two in flight; log and drop the oldest on overflow. `stop()` stops capture, calls `awaitPending(5000)`, then POSTs close. No PCM bytes are written to a file.

- [ ] **Step 5: Attach lifecycle in `DiffusionService.kt`**

Call `LiveCapture.attach(applicationContext)` in `onCreate()` and `LiveCapture.stop()` in `onDestroy()`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd android && .\gradlew.bat :app:testDebugUnitTest --tests '*LiveCaptureTest*'`
Expected: `2 passed`.

- [ ] **Step 7: Commit**

```bash
git add android/app/src/main/java/eu/dracievajko/lenswright/LiveCapture.kt \
  android/app/build.gradle.kts android/app/src/main/AndroidManifest.xml \
  android/app/src/main/java/eu/dracievajko/lenswright/DiffusionService.kt \
  android/app/src/test/java/eu/dracievajko/lenswright/LiveCaptureTest.kt
git commit -m "feat: Android live capture with mic and MediaProjection audio"
```

---

### Task 6: Android Live UI wiring

**Files:**
- Modify: `android/app/src/main/res/layout/activity_main.xml` (transcribe section ~595-758)
- Modify: `android/app/src/main/java/eu/dracievajko/lenswright/MainActivity.kt` (fields ~75-104, click listeners ~1217, permissions ~506-518)
- Test: `android/tests/test_android_live_sessions.py` (manual device checklist appended as docstring)

**Interfaces:**
- Consumes: `LiveCapture.start/stop/status`; permission request codes 2001 (`RECORD_AUDIO`) and new 2003 (`MediaProjection` consent); `DiffusionService` START path at `MainActivity.kt:518`.
- Produces: DOM ids `btnLiveStart`, `btnLiveStop`, `switchLiveMic`, `switchLiveDevice`, `liveStatus`, `liveTailMic`, `liveTailDevice`; UI helper `updateLiveUi()`.

- [ ] **Step 1: Add the Live block to the layout**

Insert after the existing Record button (`activity_main.xml:595-603`): two switches, language/model spinners, Start/Stop buttons, status text, two tail TextViews, and an "Open outputs" button. All strings English.

- [ ] **Step 2: Wire the buttons**

In `MainActivity.kt`, declare the new views, set `btnLiveStart` to auto-start the server if needed (reuse `toggleServer()` path), request `RECORD_AUDIO` (2001) when mic is enabled, and call `MediaProjectionManager.createScreenCaptureIntent()`/`startActivityForResult(..., 2003)` when device audio is enabled. On grant, POST `/v1/live/sessions` then call `LiveCapture.start(...)`. `btnLiveStop` calls `LiveCapture.stop()`, enables links and updates status.

- [ ] **Step 3: Handle permission results and status polling**

In `onRequestPermissionsResult`, add 2001 (mic) and 2003 (projection) branches that start or abort capture. Poll `LiveCapture.status()` on a UI handler every 500 ms while running and call `updateLiveUi()` to mirror status and tails.

- [ ] **Step 4: Document manual verification**

Append a docstring checklist to `android/tests/test_android_live_sessions.py`: mic-only session, device-only session (note apps that disallow playback capture produce silence), both sources, background/screen-off survival, Stop with a commit in flight, and no audio files under `output/`.

- [ ] **Step 5: Build and verify**

Run: `cd android && .\gradlew.bat assembleDebug connectedAndroidTest`
Expected: `BUILD SUCCESSFUL`; instrumented tests pass; manual checklist recorded.

- [ ] **Step 6: Commit**

```bash
git add android/app/src/main/res/layout/activity_main.xml \
  android/app/src/main/java/eu/dracievajko/lenswright/MainActivity.kt \
  android/tests/test_android_live_sessions.py
git commit -m "feat: Android Live UI for mic and device audio transcription"
```

---

### Task 7: Documentation and agent skill

**Files:**
- Modify: `docs/CHECKLIST.md`, `docs/ARCHITECTURE.md`, `pc/docs/UNIFIED-CONTRACT.md`, `skills/original/lenswright/SKILL.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: documented API rows, localhost-only note, manual test record.

- [ ] **Step 1: Update contract and architecture**

Add the four `/v1/live/*` rows to `pc/docs/UNIFIED-CONTRACT.md` with request/response shapes and the PC localhost-only note. In `docs/ARCHITECTURE.md`, add the live session flow (capture → realtime socket → append → `.txt`) and state that no audio is persisted.

- [ ] **Step 2: Update checklist and skill**

In `docs/CHECKLIST.md`, add a dated section recording the implemented endpoints, files, permissions, tests and the manual device checklist status. In `skills/original/lenswright/SKILL.md`, add Live to the capability table and endpoint list, with the separate-files and no-audio guarantees.

- [ ] **Step 3: Verify docs build clean**

Run: `cd pc && python -m pytest tests/test_webui_live.py tests/test_live_session_api.py -q` plus a markdown link check if available.
Expected: tests pass; no broken links introduced.

- [ ] **Step 4: Commit**

```bash
git add docs/CHECKLIST.md docs/ARCHITECTURE.md pc/docs/UNIFIED-CONTRACT.md skills/original/lenswright/SKILL.md
git commit -m "docs: document live transcription sessions and file-only output"
```
