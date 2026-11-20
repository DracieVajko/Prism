"""Tests for unified/gfpgan.py — GFPGAN face restoration.

Unit tests use a fake ONNX session, so no weights and no GPU are required.
The real-model test is opt-in via PRISM_RUN_MODEL_TESTS=1.
"""
import base64
import io
import os
import time
import types

import numpy as np
import pytest
from PIL import Image
from unified import gfpgan as gf

RUN_MODEL_TESTS = os.environ.get("PRISM_RUN_MODEL_TESTS") == "1"


class FakeSession:
    """Pass-through stand-in for the GFPGAN graph."""

    def __init__(self, delay: float = 0.02):
        self.delay = delay

    def get_inputs(self):
        return [types.SimpleNamespace(name="input")]

    def run(self, output_names, feeds):
        if self.delay:
            time.sleep(self.delay)
        return [np.zeros_like(feeds["input"])]


@pytest.fixture(autouse=True)
def _reset_session():
    saved = (gf._SESSION, gf._PROVIDER)
    gf._SESSION, gf._PROVIDER = None, None
    try:
        yield
    finally:
        gf._SESSION, gf._PROVIDER = saved


def _image_b64(size=(64, 48)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 170, 150)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_model_constants():
    assert gf.MODEL_ID == "gfpgan-v1.4"
    assert gf.MODEL_FILE == gf.MODEL_DIR / "gfpgan14-fp16.onnx"


def test_missing_model_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(gf, "MODEL_FILE", tmp_path / "missing.onnx")
    gf.release()
    assert gf._get_session() is None
    assert gf.is_ready() is False


def test_release_and_is_ready_with_fake_session():
    gf._SESSION = object()
    assert gf.is_ready() is True

    gf.release()
    assert gf._SESSION is None
    assert gf.is_ready() is False


def test_restore_face_b64_with_fake_session(monkeypatch):
    monkeypatch.setattr(gf, "_get_session", lambda: FakeSession())

    result = gf.restore_face_b64(_image_b64((64, 48)))

    assert result["engine"] == "gfpgan-v1.4"
    assert result["elapsed"] > 0

    out = base64.b64decode(result["image"])
    assert out[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(io.BytesIO(out)) as im:
        im.load()
        assert im.size == (64, 48)


def test_restore_face_b64_empty_raises():
    with pytest.raises(Exception):
        gf.restore_face_b64("")


@pytest.mark.integration
@pytest.mark.skipif(not RUN_MODEL_TESTS,
                    reason="set PRISM_RUN_MODEL_TESTS=1 to run model integration tests")
def test_restore_face_real_model():
    if not gf.MODEL_FILE.exists():
        pytest.skip(f"GFPGAN weights not found: {gf.MODEL_FILE}")

    try:
        result = gf.restore_face_b64(_image_b64((64, 48)))
    finally:
        gf.release()

    out = base64.b64decode(result["image"])
    with Image.open(io.BytesIO(out)) as im:
        im.load()
        assert im.size == (64, 48)
