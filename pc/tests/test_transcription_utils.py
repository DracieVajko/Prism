"""Pure formatting/chunking helpers in transcription_utils."""
import json

import unified.transcription_utils as tu


def test_timestamp_formats():
    assert tu.format_timestamp_srt(3661.25) == "01:01:01,250"
    assert tu.format_timestamp_vtt(3661.25) == "01:01:01.250"
    assert tu.format_timestamp_hms(3661.25) == "01:01:01"
    assert tu.format_timestamp_srt(-5) == "00:00:00,000"


def test_format_transcription_all_formats():
    segments = [{"start": 0.0, "end": 1.5, "text": "hello"},
                {"start": 1.5, "end": 2.0, "text": "world"}]
    txt = tu.format_transcription(segments, "txt", "en")
    assert "hello" in txt and "world" in txt
    srt = tu.format_transcription(segments, "srt", "en")
    assert "00:00:00,000 --> 00:00:01,500" in srt
    data = json.loads(tu.format_transcription(segments, "json", "en"))
    assert data["language"] == "en"
    assert len(data["segments"]) == 2
    assert tu.format_transcription(segments, "md", "en").startswith("# Transcription")


def test_split_text_respects_max_chars_and_keeps_text():
    text = "One sentence. " * 100
    chunks = tu.split_text_into_chunks(text, max_chars=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 220 for chunk in chunks)
    assert "One sentence." in chunks[0]


def test_quality_gate_passes_a_normal_transcript():
    words = [f"word{i}" for i in range(140)]
    segments = [{"start": 0.0, "end": 60.0, "text": " ".join(words)}]
    quality = tu.assess_transcript_quality(segments, duration_s=60.0)
    assert quality["ok"] is True
    assert quality["reasons"] == []
    assert quality["wpm"] == 140.0


def test_quality_gate_flags_no_speech():
    quality = tu.assess_transcript_quality([], duration_s=600.0)
    assert quality["ok"] is False
    assert quality["reasons"] == ["no-speech"]


def test_quality_gate_flags_a_decoder_loop():
    segments = [{"start": 0.0, "end": 60.0, "text": "a b c d e f " * 50}]
    quality = tu.assess_transcript_quality(segments, duration_s=60.0)
    assert quality["ok"] is False
    assert "repetition-loop" in quality["reasons"]


def test_quality_gate_flags_speech_the_vad_ate():
    segments = [{"start": 0.0, "end": 5.0, "text": "one two three four five"}]
    quality = tu.assess_transcript_quality(segments, duration_s=600.0)
    assert quality["ok"] is False
    assert "low-word-rate" in quality["reasons"]


def test_quality_gate_ignores_word_rate_on_very_short_clips():
    segments = [{"start": 0.0, "end": 2.0, "text": "hi"}]
    assert tu.assess_transcript_quality(segments, duration_s=3.0)["ok"] is True
    assert tu.assess_transcript_quality(segments)["ok"] is True
