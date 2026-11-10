"""The VAD pre-filter must degrade to full-audio transcription, never fail."""
import asyncio
from pathlib import Path

from unified_server import _transcribe_with_vad


def test_vad_without_model_returns_empty_for_fallback(monkeypatch):
    monkeypatch.setattr("unified.silero_vad.is_installed", lambda: False)
    text, lang = asyncio.run(
        _transcribe_with_vad(Path("missing.wav"), "small", "sk"))
    assert text == ""
    assert lang == "sk"


def test_vad_without_language_reports_unknown(monkeypatch):
    monkeypatch.setattr("unified.silero_vad.is_installed", lambda: False)
    text, lang = asyncio.run(
        _transcribe_with_vad(Path("missing.wav"), "small", None))
    assert text == ""
    assert lang == "unknown"


def test_vad_uses_segment_transcriber(monkeypatch):
    calls = []

    def fake_transcribe_with_vad(path, language, fn):
        calls.append(path)
        text, lang = fn("segment.wav", language)
        return text, lang, [(0.0, 1.0)]

    monkeypatch.setattr("unified.silero_vad.is_installed", lambda: True)
    monkeypatch.setattr("unified.silero_vad.transcribe_with_vad", fake_transcribe_with_vad)
    monkeypatch.setattr(
        "unified_server._transcribe_with_whisper",
        lambda path, wanted, lang, rid="": ("hello", lang or "en", []),
    )
    text, lang = asyncio.run(
        _transcribe_with_vad(Path("audio.wav"), "small", "en"))
    assert calls == ["audio.wav"]
    assert text == "hello"
    assert lang == "en"
