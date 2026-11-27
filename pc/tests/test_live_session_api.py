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
