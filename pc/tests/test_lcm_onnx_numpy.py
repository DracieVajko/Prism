"""LCM numpy pipeline helpers that do not need model weights."""
import base64
import io

import numpy as np
import pytest
from PIL import Image

from unified import lcm_onnx_numpy as lcm


def _png_b64(size=(32, 24), color=(200, 50, 50)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_find_lcm_dir_returns_none_when_absent(tmp_path):
    assert lcm.find_lcm_dir(tmp_path) is None


def test_find_lcm_dir_detects_inline_qdq(tmp_path):
    d = tmp_path / "lcm-dreamshaper-int8-onnx" / "unet"
    d.mkdir(parents=True)
    (d / "model_qdq_inline.onnx").write_bytes(b"x")
    assert lcm.find_lcm_dir(tmp_path) == tmp_path / "lcm-dreamshaper-int8-onnx"


def test_find_lcm_dir_prefers_named_asset(tmp_path):
    for name in ("LCM_Dreamshaper_ONNX", "lcm-dreamshaper-int8-onnx"):
        d = tmp_path / name / "unet"
        d.mkdir(parents=True)
        (d / "model.onnx").write_bytes(b"x")
    assert lcm.find_lcm_dir(tmp_path).name == "LCM_Dreamshaper_ONNX"


def test_b64_roundtrip_preserves_size_and_mode():
    img = lcm._b64_image_to_pil(_png_b64((40, 30)))
    assert img.size == (40, 30)
    assert img.mode == "RGB"
    again = lcm._b64_image_to_pil(lcm._pil_to_b64(img))
    assert again.size == img.size


def test_b64_rejects_garbage():
    with pytest.raises(Exception):
        lcm._b64_image_to_pil("not base64 at all!!")


def test_to_nchw_shape_and_range():
    arr = lcm._to_nchw(Image.new("RGB", (10, 10), (0, 0, 0)), 8)
    assert arr.shape == (1, 3, 8, 8)
    assert arr.min() >= -1.0 and arr.max() <= 1.0


def test_to_pil_inverts_to_nchw():
    img = Image.new("RGB", (8, 8), (10, 20, 30))
    arr = lcm._to_nchw(img, 8)
    out = lcm._to_pil(arr)
    assert out.size == (8, 8)
    assert out.mode == "RGB"
    px = np.asarray(out)[0, 0]
    assert abs(int(px[0]) - 10) <= 2 and abs(int(px[2]) - 30) <= 2


def test_colorize_changes_only_masked_region():
    src = Image.new("RGB", (16, 16), (128, 128, 128))
    mask = Image.new("L", (16, 16), 0)
    mask.paste(255, (0, 0, 8, 16))
    out = lcm._colorize(src, mask, "#ff0000")
    assert out.size == src.size
    left = np.asarray(out)[:, :8]
    right = np.asarray(out)[:, 8:]
    assert left[..., 0].mean() > left[..., 1].mean()
    assert np.allclose(right, np.asarray(src)[:, 8:])


def test_colorize_accepts_short_hex_and_mismatched_mask():
    src = Image.new("RGB", (16, 16), (90, 90, 90))
    mask = Image.new("L", (8, 8), 255)
    out = lcm._colorize(src, mask, "#0f0")
    assert out.size == src.size


def test_harmonize_without_mask_returns_png():
    out_b64 = lcm.harmonize_b64(_png_b64((24, 18)))
    img = lcm._b64_image_to_pil(out_b64)
    assert img.size == (24, 18)
