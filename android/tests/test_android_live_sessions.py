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

@pytest.mark.skip(reason="requires a device; controller implemented in Task 5")
def test_pending_commit_drained_before_close():
    """Manual device verification checklist (Task 6):

    - mic-only session produces a *-mic.txt output and no audio files.
    - device-only session (apps that disallow playback capture produce silence).
    - both sources produce separate mic/device files.
    - background/screen-off survival while running.
    - Stop with a commit in flight drains before close.
    - no audio files under output/.
    """
    # Controller-level: with a fake socket holding one in-flight commit,
    # stop() must wait for its completed event before close() is called.
    # Implement against LiveCapture.awaitPending(timeoutMs) in Task 5.
    pass
