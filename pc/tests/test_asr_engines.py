"""ASR engine helpers — install detection, env gate, language mapping."""
import pytest

from unified import nemotron_asr
from unified import sherpa_asr


# ---------------------------------------------------------------------------
# Sherpa-ONNX
# ---------------------------------------------------------------------------

def test_sherpa_model_dir_default_and_unknown():
    assert sherpa_asr._model_dir("sherpa-zipformer-small").name == "sherpa-zipformer-small"
    assert sherpa_asr._model_dir("custom-weights").name == "custom-weights"


def test_sherpa_not_installed_when_dir_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(sherpa_asr, "_model_dir", lambda model_id=None: tmp_path / "nope")
    assert sherpa_asr.is_installed() is False


def test_sherpa_installed_requires_all_parts(monkeypatch, tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    monkeypatch.setattr(sherpa_asr, "_model_dir", lambda model_id=None: d)
    (d / "encoder.onnx").write_bytes(b"x")
    (d / "decoder.onnx").write_bytes(b"x")
    (d / "joiner.onnx").write_bytes(b"x")
    assert sherpa_asr.is_installed() is False
    (d / "tokens.txt").write_text("a\n", encoding="utf-8")
    assert sherpa_asr.is_installed() is True


def test_sherpa_enabled_env_gate(monkeypatch):
    monkeypatch.setenv("PRISM_ASR_ENGINE", "sherpa")
    assert sherpa_asr.enabled() is True
    monkeypatch.setenv("PRISM_ASR_ENGINE", " Whisper ")
    assert sherpa_asr.enabled() is False
    monkeypatch.delenv("PRISM_ASR_ENGINE")
    assert sherpa_asr.enabled() is False


def test_sherpa_init_fails_cleanly_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(sherpa_asr, "_model_dir", lambda model_id=None: tmp_path / "missing")
    monkeypatch.setattr(sherpa_asr, "_recognizer", None)
    assert sherpa_asr.init("sherpa-zipformer-small") is False


def test_sherpa_language_mapping():
    assert sherpa_asr._normalise_language(None) == "auto"
    assert sherpa_asr._normalise_language("en-US") == "en"
    assert sherpa_asr._normalise_language("ZH_hans") == "zh"
    assert sherpa_asr._normalise_language("xx") == "en"


# ---------------------------------------------------------------------------
# Nemotron
# ---------------------------------------------------------------------------

def test_nemotron_not_installed_when_dir_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(nemotron_asr, "_model_dir", lambda model_id=None: tmp_path / "nope")
    assert nemotron_asr.is_installed() is False


def test_nemotron_installed_requires_weights_and_config(monkeypatch, tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    monkeypatch.setattr(nemotron_asr, "_model_dir", lambda model_id=None: d)
    (d / "config.json").write_text("{}", encoding="utf-8")
    assert nemotron_asr.is_installed() is False
    (d / "model.safetensors").write_bytes(b"x")
    assert nemotron_asr.is_installed() is True


def test_nemotron_accepts_sharded_weights(monkeypatch, tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    monkeypatch.setattr(nemotron_asr, "_model_dir", lambda model_id=None: d)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    assert nemotron_asr.is_installed() is True


def test_nemotron_enabled_env_gate(monkeypatch):
    monkeypatch.setenv("PRISM_ASR_ENGINE", "nemotron")
    assert nemotron_asr.enabled() is True
    monkeypatch.setenv("PRISM_ASR_ENGINE", "sherpa")
    assert nemotron_asr.enabled() is False


def test_nemotron_init_fails_cleanly_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(nemotron_asr, "_model_dir", lambda model_id=None: tmp_path / "missing")
    monkeypatch.setattr(nemotron_asr, "_model", None)
    assert nemotron_asr.init("nemotron-asr-0.6b") is False


def test_nemotron_language_mapping():
    assert nemotron_asr._normalise_language(None) == "auto"
    assert nemotron_asr._normalise_language("auto") == "auto"
    assert nemotron_asr._normalise_language("en-US") == "en"
    assert nemotron_asr._normalise_language("SK") == "sk"
    assert nemotron_asr._normalise_language("klingon") == "auto"
