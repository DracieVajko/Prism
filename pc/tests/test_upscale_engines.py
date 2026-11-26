"""Upscale engine dispatch: Real-ESRGAN default, SwinIR, Z-Image refine."""
import base64
import io

import pytest
from PIL import Image
from unified.upscale import upscale_image_b64


def _img_b64(size=16, color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _png_b64(size):
    return _img_b64(size, (0, 128, 255))


def test_unknown_model_is_rejected(monkeypatch):
    monkeypatch.setenv("LENSWRIGHT_UPSCALE_ENGINE", "pillow")
    with pytest.raises(ValueError, match="unknown upscale model"):
        upscale_image_b64(_img_b64(8), 2, "totally-not-a-model")


def test_swinir_dispatch(monkeypatch):
    calls = {}

    def fake_swinir(image_b64):
        calls["image"] = image_b64
        return {"image": _png_b64(64), "elapsed": 0.1,
                "engine": "swinir-realsr-x4", "scale": 4}

    monkeypatch.setattr("unified.swinir.upscale_b64", fake_swinir)
    out = upscale_image_b64(_img_b64(16), 2, "swinir-realsr-x4")
    assert out["engine"] == "swinir-realsr-x4"
    assert out["model"] == "swinir-realsr-x4"
    assert Image.open(io.BytesIO(base64.b64decode(out["image"]))).size == (32, 32)
    assert "image" in calls


def test_zimage_requires_installed_model(monkeypatch):
    monkeypatch.setenv("LENSWRIGHT_UPSCALE_ENGINE", "pillow")
    monkeypatch.setattr("unified.zimage.is_installed", lambda: False)
    with pytest.raises(RuntimeError, match="not installed"):
        upscale_image_b64(_img_b64(8), 2, "z-image-turbo")


def test_zimage_dispatch(monkeypatch):
    monkeypatch.setenv("LENSWRIGHT_UPSCALE_ENGINE", "pillow")
    seen = {}

    def fake_refine(image_b64, prompt="", strength=0.33, seed=0):
        seen.update(prompt=prompt, strength=strength, seed=seed)
        return {"image": _png_b64(32), "elapsed": 1.0,
                "engine": "zimage-turbo", "strength": strength, "steps": 9}

    monkeypatch.setattr("unified.zimage.is_installed", lambda: True)
    monkeypatch.setattr("unified.zimage.refine_b64", fake_refine)
    out = upscale_image_b64(_img_b64(16), 2, "z-image-turbo",
                            prompt="a sharp photo", strength=0.4, seed=7)
    assert out["engine"] == "zimage-turbo+realesrgan"
    assert out["base_engine"] == "pillow"
    assert out["strength"] == 0.4
    assert seen == {"prompt": "a sharp photo", "strength": 0.4, "seed": 7}
