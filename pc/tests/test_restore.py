"""Tests for unified/restore.py — InstructPix2Pix image restoration.

Unit tests use a fake pipeline and a PIL-generated image, so they need no
model weights, no torch and no GPU. The real-model test is opt-in through
PRISM_RUN_MODEL_TESTS=1.
"""
import base64
import io
import os
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest
from PIL import Image
from unified import restore as rs

PC_DIR = Path(__file__).resolve().parents[1]
RUN_MODEL_TESTS = os.environ.get("PRISM_RUN_MODEL_TESTS") == "1"


class FakePipeline:
    """Minimal stand-in for a diffusers pipeline (callable, returns .images)."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls = []

    def __call__(self, prompt, image=None, **kwargs):
        self.calls.append({"prompt": prompt, "image": image, "kwargs": kwargs})
        if self.delay:
            time.sleep(self.delay)
        return types.SimpleNamespace(images=[image.copy()])


@pytest.fixture(autouse=True)
def _reset_pipeline():
    """Keep the module-global pipeline isolated between tests."""
    saved = rs._PIPELINE
    rs._PIPELINE = None
    try:
        yield
    finally:
        rs._PIPELINE = saved


def _image(size=(96, 64)):
    img = Image.new("RGB", size, (40, 70, 110))
    for x in range(min(size)):
        img.putpixel((x, x * size[1] // size[0]), (200, 180, 60))
    return img


def _image_b64(size=(96, 64)) -> str:
    buf = io.BytesIO()
    _image(size).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_model_constants():
    assert rs.MODEL_ID == "instruct-pix2pix"
    assert rs.MODEL_DIR == PC_DIR / "models" / "instruct-pix2pix"


def test_missing_model_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "MODEL_DIR", tmp_path / "instruct-pix2pix")
    rs.release()
    assert rs._ensure_pipeline() is None
    assert rs.is_ready() is False


def test_import_without_torch_and_diffusers(tmp_path):
    """Importing restore and probing a model must not import torch/diffusers."""
    model_dir = tmp_path / "instruct-pix2pix"
    model_dir.mkdir()
    # Presence check passes; the blocked import is what makes it unavailable.
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    script = textwrap.dedent(
        """
        import importlib.abc
        import pathlib
        import sys

        BLOCKED = {"torch", "torchvision", "diffusers"}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in BLOCKED:
                    raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
                return None

        sys.meta_path.insert(0, Blocker())

        from unified import restore as rs

        assert not [m for m in sys.modules if m.split(".")[0] in BLOCKED]
        assert rs.MODEL_ID == "instruct-pix2pix"
        assert rs.is_ready() is False
        rs.MODEL_DIR = pathlib.Path(sys.argv[1])
        assert rs._ensure_pipeline() is None
        assert not [m for m in sys.modules if m.split(".")[0] in BLOCKED]
        print("OK")
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", script, str(model_dir)],
        cwd=str(PC_DIR), env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_release_and_is_ready_with_fake_pipeline():
    rs._PIPELINE = FakePipeline()
    assert rs.is_ready() is True

    rs.release()
    assert rs._PIPELINE is None
    assert rs.is_ready() is False


def test_restore_b64_with_fake_pipeline(monkeypatch, tmp_path):
    img_path = tmp_path / "input.png"
    _image((96, 64)).save(img_path)
    src = base64.b64encode(img_path.read_bytes()).decode()

    fake = FakePipeline(delay=0.02)
    monkeypatch.setattr(rs, "_ensure_pipeline", lambda: fake)

    result = rs.restore_b64(src)

    assert result["engine"] == "instruct-pix2pix"
    assert result["elapsed"] > 0

    out = base64.b64decode(result["image"])
    assert out[:8] == b"\x89PNG\r\n\x1a\n", "output should be PNG"
    with Image.open(io.BytesIO(out)) as im:
        im.load()
        assert im.format == "PNG"
        assert im.size == (96, 64), f"size mismatch: {im.size}"

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["image"].size == (512, 512)
    assert "restore" in call["prompt"].lower()
    assert call["kwargs"]["num_inference_steps"] == 20
    assert call["kwargs"]["guidance_scale"] == 7.5
    assert call["kwargs"]["image_guidance_scale"] == 1.5


def test_restore_b64_without_model_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "MODEL_DIR", tmp_path / "instruct-pix2pix")
    with pytest.raises(RuntimeError, match="not loaded"):
        rs.restore_b64(_image_b64((16, 16)))


def test_restore_b64_empty_raises():
    with pytest.raises(Exception):
        rs.restore_b64("")


def test_restore_api_contract(monkeypatch):
    """The endpoint keeps its contract: 400 bad input, 503 model unavailable."""
    import unified_server as us
    from fastapi.testclient import TestClient

    client = TestClient(us.app)

    r = client.post("/v1/images/restore", json={})
    assert r.status_code == 400, r.text[:200]
    assert r.json()["error"] == "missing image b64"

    monkeypatch.setattr(rs, "_ensure_pipeline", lambda: None)
    r = client.post("/v1/images/restore", json={"image": _image_b64((16, 16))})
    assert r.status_code == 503, r.text[:200]
    assert r.json()["error"] == "will not load"


@pytest.mark.integration
@pytest.mark.skipif(not RUN_MODEL_TESTS,
                    reason="set PRISM_RUN_MODEL_TESTS=1 to run model integration tests")
def test_restore_real_model():
    if not (rs.MODEL_DIR / "model_index.json").exists():
        pytest.skip(f"InstructPix2Pix weights not found: {rs.MODEL_DIR}")
    result = rs.restore_b64(_image_b64((64, 64)), steps=2)
    assert result["engine"] == "instruct-pix2pix"
    out = base64.b64decode(result["image"])
    with Image.open(io.BytesIO(out)) as im:
        im.load()
        assert im.size == (64, 64)
    rs.release()
