"""Tests for unified/restore.py — InstructPix2Pix image restoration."""
import base64
from pathlib import Path

import pytest

from unified import restore as rs


def test_model_constants():
    assert rs.MODEL_ID == "instruct-pix2pix"
    assert rs.MODEL_DIR.exists(), "InstructPix2Pix model not downloaded"


def test_restore_b64_returns_image():
    img_path = Path(r"C:\Users\DracieVajko\Pictures\Camera\IMG_20260927_140020.jpg")
    if not img_path.exists():
        pytest.skip("test image not available")
    raw = img_path.read_bytes()
    b64 = base64.b64encode(raw).decode()
    result = rs.restore_b64(b64)
    assert "image" in result
    assert result["engine"] == "instruct-pix2pix"
    assert "elapsed" in result
    assert result["elapsed"] > 0
    # Output should be valid PNG
    out = base64.b64decode(result["image"])
    assert out[:4] == b"\x89PNG", "output should be PNG"
    from PIL import Image
    import io
    im = Image.open(io.BytesIO(out))
    orig = Image.open(img_path)
    assert im.size == orig.size, f"size mismatch: {im.size} vs {orig.size}"


def test_restore_b64_empty_raises():
    with pytest.raises(Exception):
        rs.restore_b64("")


def test_release_and_is_ready():
    rs.release()
    assert rs.is_ready() is False
    session = rs._ensure_pipeline()
    assert session is not None
    assert rs.is_ready() is True
    rs.release()
