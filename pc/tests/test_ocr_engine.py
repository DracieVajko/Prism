"""OCR/VLM engine mechanics — no weights, no GPU.

The generation path in `unified/ocr_engine.py` is shared by `/v1/images/ocr`,
`/v1/images/caption` and vision chat, and it is the path that once shipped a
shadowed `import torch` (UnboundLocalError, masked as an empty 200 response).
These tests run `generate_pil` against a fake model so a regression fails here
instead of in production.
"""
import types

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from unified import ocr_engine as oc  # noqa: E402


class FakeProcessor:
    def __init__(self):
        self.decoded = None

    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, tokens, skip_special_tokens=True):
        return "  OCR TEXT  "


class FakeModel:
    """Minimal stand-in for a transformers image-text model."""

    def __init__(self, fail_times: int = 0, reply: str = "OCR TEXT"):
        self.device = torch.device("cpu")
        self.fail_times = fail_times
        self.calls = 0
        self.params = [types.SimpleNamespace(device=torch.device("cpu"))]

    def eval(self):
        return self

    def parameters(self):
        return iter(self.params)

    def generate(self, **inputs):
        self.calls += 1
        if self.fail_times >= self.calls:
            raise RuntimeError("transient generate failure")
        return torch.tensor([[1, 2, 3, 9, 9]])


def _engine(model=None, processor=None) -> oc.OCREngine:
    eng = oc.OCREngine("glm-ocr")
    eng.model = model or FakeModel()
    eng.processor = processor or FakeProcessor()
    eng._loaded = True
    return eng


def _image():
    return Image.new("RGB", (8, 8), (255, 255, 255))


def test_generate_pil_returns_decoded_text():
    """Regression guard: the torch binding must resolve inside _run_once."""
    eng = _engine()
    assert eng.generate_pil(_image(), "Text Recognition:") == "OCR TEXT"


def test_generate_pil_counts_in_flight_and_touches_last_used():
    eng = _engine()
    eng.last_used = 0.0
    eng.generate_pil(_image(), "prompt")
    assert eng.in_flight == 0
    assert eng.last_used > 0.0


def test_generate_pil_retries_in_place_without_reloading(monkeypatch):
    eng = _engine(FakeModel(fail_times=1))
    called = {"init": 0, "release": 0}
    monkeypatch.setattr(eng, "init", lambda: called.__setitem__("init", called["init"] + 1) or True)
    monkeypatch.setattr(eng, "release", lambda *a, **k: called.__setitem__("release", called["release"] + 1))
    assert eng.generate_pil(_image(), "prompt") == "OCR TEXT"
    assert eng.model.calls == 2          # failed once, retried once
    assert called == {"init": 0, "release": 0}, "a transient error must not reload the model"


def test_generate_pil_gives_up_after_two_attempts(monkeypatch):
    """A deterministic error must not trigger a multi-minute reload."""
    eng = _engine(FakeModel(fail_times=99))
    monkeypatch.setattr(eng, "init", lambda: pytest.fail("must not reload by default"))
    monkeypatch.setattr(eng, "release", lambda *a, **k: pytest.fail("must not release by default"))
    with pytest.raises(RuntimeError, match="transient generate failure"):
        eng.generate_pil(_image(), "prompt")
    assert eng.model.calls == 2


def test_generate_pil_reloads_when_explicitly_enabled(monkeypatch):
    eng = _engine(FakeModel(fail_times=2))
    monkeypatch.setenv("LENSWRIGHT_OCR_RELOAD_ON_ERROR", "1")
    reloaded = {"n": 0}

    def fake_init():
        reloaded["n"] += 1
        eng.model = FakeModel()   # fresh, working model after the reload
        return True

    monkeypatch.setattr(eng, "init", fake_init)
    monkeypatch.setattr(eng, "release", lambda *a, **k: True)
    assert eng.generate_pil(_image(), "prompt") == "OCR TEXT"
    assert reloaded["n"] == 1


def test_idle_helpers_ignore_released_engines(monkeypatch):
    eng = _engine()
    monkeypatch.setattr(oc, "_engines", {"glm-ocr": eng})
    eng.last_used = 1000.0
    assert oc.ocr_resident() == ["glm-ocr"]
    assert oc.ocr_in_flight() == 0
    assert oc.ocr_idle_seconds() is not None

    eng._loaded = False
    eng.model = None
    assert oc.ocr_resident() == []
    assert oc.ocr_idle_seconds() is None


def test_ocr_ready_reflects_loaded_weights(monkeypatch):
    eng = _engine()
    monkeypatch.setattr(oc, "_engines", {"glm-ocr": eng})
    monkeypatch.setattr(oc, "resolve_ocr_model", lambda m: "glm-ocr")
    assert oc.ocr_ready("glm-ocr") is True
    eng._loaded = False
    assert oc.ocr_ready("glm-ocr") is False


def test_free_ram_is_measured():
    free = oc._free_ram_gb()
    assert free is None or free > 0
    assert oc.load_ram_requirement_gb("glm-ocr") > 2.6   # 2.66 GB of weights


def test_init_refuses_instead_of_crashing_when_ram_is_short(monkeypatch):
    """Loading materialises ~2x the checkpoint; on a full box torch_cpu.dll
    used to fault and take the whole server down with it."""
    eng = oc.OCREngine("glm-ocr")
    monkeypatch.setattr(eng, "is_installed", lambda: True)
    monkeypatch.setattr(oc, "_free_ram_gb", lambda: 1.0)
    monkeypatch.setattr(oc.AutoProcessor, "from_pretrained",
                        lambda *a, **k: pytest.fail("must not start loading"))
    assert eng.init() is False
    assert "not enough free RAM" in eng.last_error
    assert eng.model is None and eng.processor is None


def test_init_skips_the_guard_when_ram_is_unknown(monkeypatch):
    eng = oc.OCREngine("glm-ocr")
    monkeypatch.setattr(eng, "is_installed", lambda: True)
    monkeypatch.setattr(oc, "_free_ram_gb", lambda: None)
    monkeypatch.setattr(oc.AutoProcessor, "from_pretrained",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop here")))
    assert eng.init() is False
    assert "not enough free RAM" not in eng.last_error
