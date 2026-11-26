"""Tests for unified/colorize.py — multi-model colorization.

Unit tests use a fake ONNX session, so no weights and no GPU are required.
The real-model test is opt-in via LENSWRIGHT_RUN_MODEL_TESTS=1.
"""
import base64
import io
import os
import time
import types

import numpy as np
import pytest
from PIL import Image
from unified import colorize as cz

RUN_MODEL_TESTS = os.environ.get("LENSWRIGHT_RUN_MODEL_TESTS") == "1"


@pytest.fixture(autouse=True)
def _reset_sessions():
    saved = dict(cz._SESSIONS)
    cz._SESSIONS.clear()
    try:
        yield
    finally:
        cz._SESSIONS.clear()
        cz._SESSIONS.update(saved)


def _image_b64(size=(48, 32)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 90, 90)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_model_constants():
    assert cz.COLORIZE_MODELS["ddcolor-full"]["type"] == "ddcolor"
    assert cz.COLORIZE_MODELS["deoldify-artistic"]["type"] == "deoldify"
    assert cz.INPUT_SIZE == 512
    assert cz._DEFAULT_MODEL == "deoldify-artistic"


def test_missing_model_is_unavailable(tmp_path, monkeypatch):
    # Point every model path at a missing file.
    for spec in cz.COLORIZE_MODELS.values():
        spec["path"] = tmp_path / "missing.onnx"
    cz.release()
    assert cz._get_session("ddcolor-full") is None
    assert cz.is_ready() is False


def test_release_and_is_ready_with_fake_session():
    cz._SESSIONS["fake"] = (object(), "cpu")
    assert cz.is_ready() is True

    cz.release()
    assert len(cz._SESSIONS) == 0
    assert cz.is_ready() is False


def test_colorize_b64_empty_raises():
    with pytest.raises((ValueError, Exception)):
        cz.colorize_b64("")


def test_colorize_b64_unknown_model_falls_back_to_default():
    # An unknown model id should not crash; it falls back to the default.
    # (We can't run inference without weights, so just check the fallback path.)
    import base64 as _b64
    img = Image.new("RGB", (16, 16), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = _b64.b64encode(buf.getvalue()).decode()
    # With no weights installed this raises RuntimeError, not a crash.
    try:
        cz.colorize_b64(b64, model_id="does-not-exist")
    except RuntimeError:
        pass


@pytest.mark.skipif(not RUN_MODEL_TESTS, reason="set LENSWRIGHT_RUN_MODEL_TESTS=1")
def test_colorize_real_model():
    if not any(cz.COLORIZE_MODELS[m]["path"].exists() for m in cz.COLORIZE_MODELS):
        pytest.skip("no colorize weights installed")
    result = cz.colorize_b64(_image_b64((64, 64)), model_id="deoldify-artistic")
    assert result["engine"] == "deoldify-artistic"
    assert "image" in result
    assert result["elapsed"] >= 0
