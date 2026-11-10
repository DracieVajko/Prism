"""CLI helper tests: benchmark image stats and base64 output saving."""
import base64
import io

from PIL import Image

import benchmark
import generate_image


def _png_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def test_stats_on_black_image(tmp_path):
    p = tmp_path / "black.png"
    p.write_bytes(_png_bytes((0, 0, 0)))
    out = benchmark.stats(str(p))
    assert out["mean"] == 0.0
    assert out["black_pct"] == 100.0


def test_stats_on_uniform_image(tmp_path):
    p = tmp_path / "grey.png"
    p.write_bytes(_png_bytes((128, 128, 128)))
    out = benchmark.stats(str(p))
    assert out["mean"] == 128.0
    assert out["std"] == 0.0
    assert out["black_pct"] == 0.0


def test_benchmark_jobs_are_well_formed():
    assert benchmark.JOBS
    for model, size in benchmark.JOBS:
        assert isinstance(model, str) and model
        assert "x" in size


def test_save_b64_writes_exact_bytes(tmp_path):
    raw = _png_bytes((10, 20, 30))
    out = tmp_path / "img.png"
    generate_image.save_b64(base64.b64encode(raw).decode(), str(out))
    assert out.read_bytes() == raw


def test_save_b64_roundtrips_payload_bytes(tmp_path):
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    out = tmp_path / "sig.png"
    generate_image.save_b64(base64.b64encode(raw).decode(), str(out))
    assert out.read_bytes() == raw
