"""Tests for unified/swinir.py — SwinIR super-resolution.

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

from unified import swinir as sw

RUN_MODEL_TESTS = os.environ.get("PRISM_RUN_MODEL_TESTS") == "1"


class FakeSession:
    """4x nearest-neighbour upscaler standing in for the SwinIR graph."""

    def __init__(self, delay: float = 0.02):
        self.delay = delay

    def get_inputs(self):
        return [types.SimpleNamespace(name="input")]

    def run(self, output_names, feeds):
        if self.delay:
            time.sleep(self.delay)
        inp = feeds["input"]
        return [np.repeat(np.repeat(inp, 4, axis=2), 4, axis=3)]


@pytest.fixture(autouse=True)
def _reset_session():
    saved = (sw._SESSION, sw._PROVIDER)
    sw._SESSION, sw._PROVIDER = None, None
    try:
        yield
    finally:
        sw._SESSION, sw._PROVIDER = saved


def _image_b64(size=(64, 64)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_model_constants():
    assert sw.MODEL_ID == "swinir-realsr-x4"
    assert sw.MODEL_FILE == sw.MODEL_DIR / "swinir_realsr_x4.onnx"


def test_missing_model_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(sw, "MODEL_FILE", tmp_path / "missing.onnx")
    sw.release()
    assert sw._get_session() is None
    assert sw.is_ready() is False


def test_release_and_is_ready_with_fake_session():
    sw._SESSION = object()
    assert sw.is_ready() is True

    sw.release()
    assert sw._SESSION is None
    assert sw.is_ready() is False


def test_upscale_b64_with_fake_session(monkeypatch):
    monkeypatch.setattr(sw, "_get_session", lambda: FakeSession())

    result = sw.upscale_b64(_image_b64((64, 64)))

    assert result["engine"] == "swinir-realsr-x4"
    assert result["scale"] == 4
    assert result["elapsed"] > 0

    out = base64.b64decode(result["image"])
    assert out[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(io.BytesIO(out)) as im:
        im.load()
        assert im.size == (256, 256)


def test_upscale_b64_empty_raises():
    with pytest.raises(Exception):
        sw.upscale_b64("")


@pytest.mark.integration
@pytest.mark.skipif(not RUN_MODEL_TESTS,
                    reason="set PRISM_RUN_MODEL_TESTS=1 to run model integration tests")
def test_upscale_real_model():
    if not sw.MODEL_FILE.exists():
        pytest.skip(f"SwinIR weights not found: {sw.MODEL_FILE}")

    try:
        result = sw.upscale_b64(_image_b64((64, 64)))
    finally:
        sw.release()

    out = base64.b64decode(result["image"])
    with Image.open(io.BytesIO(out)) as im:
        im.load()
        assert im.size == (256, 256)
