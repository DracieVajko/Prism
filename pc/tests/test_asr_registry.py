"""ASR engine registry: model-name resolution and delegation."""
import types
import wave

from unified import asr, nemotron_asr, sherpa_asr


def test_explicit_model_selects_engine():
    engine, model_id = asr.resolve_engine("nemotron-asr-0.6b")
    assert engine is not None and engine.module is nemotron_asr
    assert model_id == "nemotron-asr-0.6b"

    engine, model_id = asr.resolve_engine("sherpa-zipformer-small")
    assert engine is not None and engine.module is sherpa_asr
    assert model_id == "sherpa-zipformer-small"


def test_whisper_alias_is_turbo_when_no_engine_enabled(monkeypatch):
    monkeypatch.delenv("PRISM_ASR_ENGINE", raising=False)
    engine, model_id = asr.resolve_engine("whisper-1")
    assert engine is None
    assert model_id == "turbo"


def test_env_gate_selects_engine_for_alias(monkeypatch):
    monkeypatch.setenv("PRISM_ASR_ENGINE", "sherpa")
    engine, model_id = asr.resolve_engine("whisper-1")
    assert engine is not None and engine.module is sherpa_asr
    assert model_id == sherpa_asr.MODEL_ID


def test_unknown_model_passes_through():
    engine, model_id = asr.resolve_engine("custom-whisper")
    assert engine is None
    assert model_id == "custom-whisper"


def test_engine_delegates_install_and_transcribe(monkeypatch):
    engine = asr.AsrEngine("fake", "fake", "hint", nemotron_asr)
    monkeypatch.setattr(nemotron_asr, "is_installed", lambda mid=None: mid == "fake-1")
    monkeypatch.setattr(
        nemotron_asr, "transcribe",
        lambda path, lang, mid: (f"{path}|{lang}|{mid}", "en"))
    assert engine.installed("fake-1") is True
    assert engine.transcribe("a", "sk", "fake-1") == ("a|sk|fake-1", "en")


class _CumulativeStream:
    """Sherpa's result.text is cumulative; it must not be appended per chunk."""

    def __init__(self):
        self.text = ""

    def accept_waveform(self, sample_rate, chunk):
        self.text += "x"

    def is_ready(self):
        return False

    def decode(self):
        pass

    def input_finished(self):
        self.text += "!"

    @property
    def result(self):
        return types.SimpleNamespace(text=self.text)


def test_sherpa_reads_cumulative_text_once(monkeypatch, tmp_path):
    wav = tmp_path / "tone.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 3200)

    stream = _CumulativeStream()
    monkeypatch.setattr(
        sherpa_asr, "_recognizer",
        types.SimpleNamespace(create_stream=lambda: stream))
    text, lang = sherpa_asr.transcribe(
        str(wav), "en", sherpa_asr.MODEL_ID)
    assert text == "x" * 2 + "!"
    assert lang == "en"
