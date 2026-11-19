"""Tests for unified/colorize.py — DDColor colorization.

Unit tests never load the ONNX weights; the real-model test is opt-in via
PRISM_RUN_MODEL_TESTS=1.
"""
import base64
import io
import os

import pytest
from PIL import Image

from unified import colorize as cz

RUN_MODEL_TESTS = os.environ.get("PRISM_RUN_MODEL_TESTS") == "1"


@pytest.fixture(autouse=True)
def _reset_session():
    saved = (cz._SESSION, cz._PROVIDER, cz._ACTIVE_ID)
    cz._SESSION, cz._PROVIDER = None, None
    try:
        yield
    finally:
        cz._SESSION, cz._PROVIDER, cz._ACTIVE_ID = saved


def _image_b64(size=(48, 32)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 90, 90)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_model_constants():
    assert cz.MODEL_ID == "ddcolor-full"
    assert cz.INPUT_SIZE == 512
    assert cz.MODEL_FILE == cz.MODEL_DIR / "ddcolor-512-fp16.onnx"


def test_missing_model_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(cz, "MODEL_FILE", tmp_path / "missing-full.onnx")
    monkeypatch.setattr(cz, "FALLBACK_FILE", tmp_path / "missing-tiny.onnx")
    cz.release()
    assert cz._get_session() is None
    assert cz.is_ready() is False


def test_release_and_is_ready_with_fake_session():
    cz._SESSION = object()
    assert cz.is_ready() is True

    cz.release()
    assert cz._SESSION is None
    assert cz.is_ready() is False


def test_colorize_b64_empty_raises():
    with pytest.raises(Exception):
        cz.colorize_b64("")


@pytest.mark.integration
@pytest.mark.skipif(not RUN_MODEL_TESTS,
                    reason="set PRISM_RUN_MODEL_TESTS=1 to run model integration tests")
def test_colorize_real_model():
    if not cz.MODEL_FILE.exists():
        pytest.skip(f"DDColor weights not found: {cz.MODEL_FILE}")
    if cz.cv2 is None:
        pytest.skip("opencv not installed")

    try:
        result = cz.colorize_b64(_image_b64())
    finally:
        cz.release()

    assert result["engine"] in ("ddcolor-full", "ddcolor-tiny")
    out = base64.b64decode(result["image"])
    assert out[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(io.BytesIO(out)) as im:
        im.load()
        assert im.size == (48, 32)
