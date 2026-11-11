"""Tests for unified/transcription_utils.py — output formatting, chunking,
validation, duration/estimate helpers."""
import json
from pathlib import Path

import pytest

from unified import transcription_utils as tu

SEGS = [
    {"text": "hello world", "start": 0.0, "end": 2.5},
    {"text": "second line", "start": 3.0, "end": 5.5},
]


# ---------------- timestamps ----------------
def test_format_timestamp_srt():
    assert tu.format_timestamp_srt(0) == "00:00:00,000"
    assert tu.format_timestamp_srt(3723.456) == "01:02:03,456"


def test_format_timestamp_vtt():
    assert tu.format_timestamp_vtt(0) == "00:00:00.000"
    assert tu.format_timestamp_vtt(61.5) == "00:01:01.500"


def test_format_timestamp_hms():
    assert tu.format_timestamp_hms(0) == "00:00:00"
    assert tu.format_timestamp_hms(3661) == "01:01:01"


# ---------------- format_as_* ----------------
def test_format_as_txt():
    out = tu.format_as_txt(SEGS, "sk")
    assert "# Transcription (Language: sk)" in out
    assert "[001] hello world" in out
    assert "[002] second line" in out


def test_format_as_md():
    out = tu.format_as_md(SEGS, "en")
    assert "# Transcription" in out
    assert "### [001] 00:00:00 - 00:00:02" in out


def test_format_as_srt():
    out = tu.format_as_srt(SEGS)
    lines = out.split("\n")
    assert lines[0] == "1"
    assert "00:00:00,000 --> 00:00:02,500" in lines[1]
    assert lines[2] == "hello world"


def test_format_as_json():
    out = tu.format_as_json(SEGS, "cs")
    data = json.loads(out)
    assert data["language"] == "cs"
    assert data["segment_count"] == 2


# ---------------- dispatcher ----------------
def test_format_transcription_dispatch():
    assert tu.format_transcription([], "txt", "en") == tu.format_as_txt([], "en")
    assert tu.format_transcription([], "srt", "en") == tu.format_as_srt([], "en")
    assert tu.format_transcription([], "md", "en") == tu.format_as_md([], "en")
    assert tu.format_transcription([], "json", "en") == tu.format_as_json([], "en")
    with pytest.raises(ValueError):
        tu.format_transcription([], "unknown_format")


# ---------------- chunking ----------------
def test_split_text_into_chunks_basic():
    text = "a " * 5000
    chunks = tu.split_text_into_chunks(text, max_chars=1000, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 1000


def test_split_text_into_chunks_short():
    assert tu.split_text_into_chunks("short", max_chars=1000) == ["short"]
    assert tu.split_text_into_chunks("") == []


def test_split_text_by_sentences():
    chunks = tu.split_text_by_sentences("First sentence. Second sentence! Third?", max_chars=20)
    assert chunks == ["First sentence.", "Second sentence!", "Third?"]


# ---------------- validation ----------------
def test_validate_audio_file(tmp_path):
    good = tmp_path / "x.mp3"; good.write_bytes(b"x")
    assert tu.validate_audio_file(str(good))[0] is True
    bad = tmp_path / "x.xyz"; bad.write_bytes(b"x")
    assert tu.validate_audio_file(str(bad))[0] is False
    missing = tmp_path / "nope.mp3"
    assert tu.validate_audio_file(str(missing))[0] is False


def test_validate_video_file(tmp_path):
    good = tmp_path / "x.mp4"; good.write_bytes(b"x")
    assert tu.validate_video_file(str(good))[0] is True
    bad = tmp_path / "x.xyz"; bad.write_bytes(b"x")
    assert tu.validate_video_file(str(bad))[0] is False


# ---------------- estimate ----------------
def test_estimate_transcription_time():
    assert tu.estimate_transcription_time(100, "whisper") == 30.0
    assert tu.estimate_transcription_time(100, "sherpa") == 10.0
    assert tu.estimate_transcription_time(100, "nemotron") == 50.0
    assert tu.estimate_transcription_time(100, "unknown") == 30.0


# ---------------- merge ----------------
def test_merge_transcript_segments_offsets():
    segs1 = [{"text": "a", "start": 0, "end": 2}]
    segs2 = [{"text": "b", "start": 0, "end": 3}]
    merged = tu.merge_transcript_segments([segs1, segs2])
    assert merged[0]["start"] == 0 and merged[0]["end"] == 2
    assert merged[1]["start"] == 2 and merged[1]["end"] == 5


# ---------------- get_media_duration_seconds (mocks ffprobe) ----------------
def test_get_media_duration_seconds_ffprobe(monkeypatch, tmp_path):
    f = tmp_path / "audio.wav"; f.write_bytes(b"x")
    import unified.audio_splitter as asplitter
    monkeypatch.setattr(asplitter, "get_audio_duration", lambda p: 42.5)
    assert tu.get_media_duration_seconds(str(f)) == 42.5
