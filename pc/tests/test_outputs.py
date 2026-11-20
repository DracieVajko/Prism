"""Output path validation: long fuzz paths must be 400, never 500."""
import pytest
import unified_server as us
from fastapi.testclient import TestClient


def test_output_path_accepts_normal_relative_path(tmp_path, monkeypatch):
    monkeypatch.setattr(us, "OUTPUT_DIR", tmp_path)
    (tmp_path / "text").mkdir()
    target = tmp_path / "text" / "note.txt"
    target.write_text("hello", encoding="utf-8")
    assert us._output_path("text/note.txt") == target.resolve()


@pytest.mark.parametrize("rel", [
    "text/" + "a" * 300,                 # one segment over 255 bytes
    "/".join(["ab"] * 600),              # total path over 1024 bytes
    "text//note.txt",                    # empty segment
    "text/../note.txt",                  # traversal segment
    "",                                  # empty path
])
def test_output_path_rejects_invalid_fuzz_paths(rel, tmp_path, monkeypatch):
    monkeypatch.setattr(us, "OUTPUT_DIR", tmp_path)
    with pytest.raises(ValueError):
        us._output_path(rel)


def test_delete_long_output_is_400_not_500(tmp_path, monkeypatch):
    monkeypatch.setattr(us, "OUTPUT_DIR", tmp_path)
    c = TestClient(us.app)
    rel = "text/" + "a" * 300
    r = c.delete(f"/v1/outputs/{rel}")
    assert r.status_code == 400, r.text[:200]
    assert "too long" in r.json()["error"]


def test_get_long_output_is_400_not_500(tmp_path, monkeypatch):
    monkeypatch.setattr(us, "OUTPUT_DIR", tmp_path)
    c = TestClient(us.app)
    rel = "text/" + "a" * 300
    r = c.get(f"/v1/outputs/{rel}")
    assert r.status_code == 400, r.text[:200]


def test_output_get_and_delete_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(us, "OUTPUT_DIR", tmp_path)
    (tmp_path / "text").mkdir()
    target = tmp_path / "text" / "note.txt"
    target.write_text("hello", encoding="utf-8")
    c = TestClient(us.app)
    r = c.get("/v1/outputs/text/note.txt")
    assert r.status_code == 200
    assert r.text == "hello"
    r = c.delete("/v1/outputs/text/note.txt")
    assert r.status_code == 200
    assert not target.exists()
