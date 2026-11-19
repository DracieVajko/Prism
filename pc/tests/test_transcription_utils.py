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
