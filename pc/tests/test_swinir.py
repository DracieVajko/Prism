"""Tests for unified/swinir.py — SwinIR super-resolution."""
import base64
from pathlib import Path

import pytest

from unified import swinir as sw


def test_model_constants():
    assert sw.MODEL_ID == "swinir-realsr-x4"
    assert sw.MODEL_FILE.exists(), "SwinIR model not downloaded"


def test_upscale_b64_returns_image():
    img_path = Path(r"C:\Users\DracieVajko\Pictures\Camera\IMG_20260927_140020.jpg")
    if not img_path.exists():
        pytest.skip("test image not available")
    # Use a small crop (64x64) for the test
    from PIL import Image
    im = Image.open(img_path).resize((64, 64))
    buf = __import__("io").BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    result = sw.upscale_b64(b64)
    assert "image" in result
    assert result["engine"] == "swinir-realsr-x4"
    assert "elapsed" in result
    assert result["elapsed"] > 0
    out = base64.b64decode(result["image"])
    assert out[:4] == b"\x89PNG", "output should be PNG"
    from PIL import Image as PImage
    import io
    im_out = PImage.open(io.BytesIO(out))
    assert im_out.size == (256, 256), f"expected 256x256, got {im_out.size}"


def test_upscale_b64_empty_raises():
    with pytest.raises(Exception):
        sw.upscale_b64("")


def test_release_and_is_ready():
    sw.release()
    assert sw.is_ready() is False
    session = sw._get_session()
    assert session is not None
    assert sw.is_ready() is True
    sw.release()
