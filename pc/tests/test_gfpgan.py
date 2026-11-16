"""Tests for unified/gfpgan.py — GFPGAN face restoration."""
import base64
from pathlib import Path

import pytest

from unified import gfpgan as gf


def test_model_constants():
    assert gf.MODEL_ID == "gfpgan-v1.4"
    assert gf.MODEL_FILE.exists(), "GFPGAN model not downloaded"


def test_restore_face_b64_returns_image():
    img_path = Path(r"C:\Users\DracieVajko\Pictures\Camera\IMG_20260927_140020.jpg")
    if not img_path.exists():
        pytest.skip("test image not available")
    # Use a 512x512 face-like crop
    from PIL import Image
    im = Image.open(img_path).resize((512, 512))
    buf = __import__("io").BytesIO()
    im.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    result = gf.restore_face_b64(b64)
    assert "image" in result
    assert result["engine"] == "gfpgan-v1.4"
    assert "elapsed" in result
    assert result["elapsed"] > 0
    out = base64.b64decode(result["image"])
    assert out[:4] == b"\x89PNG", "output should be PNG"
    from PIL import Image as PImage
    import io
    im_out = PImage.open(io.BytesIO(out))
    assert im_out.size == (512, 512), f"expected 512x512, got {im_out.size}"


def test_restore_face_b64_empty_raises():
    with pytest.raises(Exception):
        gf.restore_face_b64("")


def test_release_and_is_ready():
    gf.release()
    assert gf.is_ready() is False
    session = gf._get_session()
    assert session is not None
    assert gf.is_ready() is True
    gf.release()
