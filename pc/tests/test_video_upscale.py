"""Video upscale input validation (no model/ffmpeg work needed)."""
import unified_server as us
from fastapi.testclient import TestClient


def test_video_upscale_requires_file():
    c = TestClient(us.app)
    r = c.post("/v1/videos/upscale", data={"scale": "2"})
    assert r.status_code == 400
    assert r.json()["error"] == "missing file"


def test_video_upscale_rejects_bad_scale():
    c = TestClient(us.app)
    r = c.post("/v1/videos/upscale",
               files={"file": ("x.mp4", b"x", "video/mp4")},
               data={"scale": "9"})
    assert r.status_code == 400
    assert "scale" in r.json()["error"]


def test_video_upscale_rejects_unknown_model():
    c = TestClient(us.app)
    r = c.post("/v1/videos/upscale",
               files={"file": ("x.mp4", b"x", "video/mp4")},
               data={"scale": "2", "model": "not-a-model"})
    assert r.status_code == 400
    assert "unknown upscale model" in r.json()["error"]
