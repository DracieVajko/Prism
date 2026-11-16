"""Tests for unified/colorize.py — DDColor colorization."""
import base64
from pathlib import Path

import pytest

from unified import colorize as cz


def test_model_constants():
    assert cz.MODEL_ID == "ddcolor-tiny"
    assert cz.INPUT_SIZE == 512
    assert cz.MODEL_FILE.exists(), "DDColor model not downloaded"


def test_colorize_b64_returns_image():
    img_path = Path(r"C:\Users\DracieVajko\Pictures\Camera\IMG_20260927_140020.jpg")
    if not img_path.exists():
        pytest.skip("test image not available")
    raw = img_path.read_bytes()
    b64 = base64.b64encode(raw).decode()
    result = cz.colorize_b64(b64)
    assert "image" in result
    assert result["engine"] == "ddcolor-tiny"
    assert "elapsed" in result
    assert result["elapsed"] > 0
    # Output should be valid base64 PNG
    out = base64.b64decode(result["image"])
    assert out[:4] == b"\x89PNG", "output should be PNG"
    # Output size should match input size (DDColor preserves resolution)
    from PIL import Image
    import io
    im = Image.open(io.BytesIO(out))
    orig = Image.open(img_path)
    assert im.size == orig.size, f"size mismatch: {im.size} vs {orig.size}"


def test_colorize_b64_empty_raises():
    with pytest.raises(Exception):
        cz.colorize_b64("")


def test_release_and_is_ready():
    cz.release()
    assert cz.is_ready() is False
    # Re-loading should work
    session = cz._get_session()
    assert session is not None
    assert cz.is_ready() is True
    cz.release()
