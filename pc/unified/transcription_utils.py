"""
Transcription utilities for Lenswright PC server.

Provides:
- Multi-format output formatting (txt, md, srt, json)
- Text chunking for AI context limits
- Multi-file batch transcription orchestration
- Video/audio file validation and preparation
"""

import logging
import re
from datetime import timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum characters per text chunk for AI context limits
DEFAULT_MAX_CHARS_PER_CHUNK = 8000

# Supported output formats
SUPPORTED_FORMATS = ("txt", "md", "srt", "json")

# Quality gates for an ASR result. A transcript below any of these is treated
# as broken and worth one more decoding attempt with stronger settings.
MIN_WORDS_PER_MINUTE = 20.0
MIN_UNIQUE_WORD_RATIO = 0.30
MIN_WORDS_FOR_RATIO = 40
LOOP_GRAM_WORDS = 6
LOOP_MIN_REPEATS = 8
LOOP_MIN_GRAMS = 20


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    td = timedelta(seconds=max(0, seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    td = timedelta(seconds=max(0, seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def format_timestamp_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS"""
    td = timedelta(seconds=max(0, seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_as_txt(segments: list[dict], language: str = "unknown") -> str:
    """Format transcription as plain text.

    Args:
        segments: List of dicts with 'text', 'start', 'end' keys
        language: Detected language code

    Returns:
        Plain text transcription
    """
    lines = [f"# Transcription (Language: {language})"]
    lines.append("=" * 60)
    lines.append("")

    for i, seg in enumerate(segments, 1):
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{i:03d}] {text}")
            lines.append("")

    return "\n".join(lines)


def format_as_md(segments: list[dict], language: str = "unknown") -> str:
    """Format transcription as Markdown with timestamps.

    Args:
        segments: List of dicts with 'text', 'start', 'end' keys
        language: Detected language code

    Returns:
        Markdown-formatted transcription
    """
    lines = ["# Transcription"]
    lines.append(f"**Language:** {language}")
    lines.append(f"**Total segments:** {len(segments)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, seg in enumerate(segments, 1):
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        if text:
            start_str = format_timestamp_hms(start)
            end_str = format_timestamp_hms(end)
            lines.append(f"### [{i:03d}] {start_str} - {end_str}")
            lines.append("")
            lines.append(text)
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def format_as_srt(segments: list[dict], language: str = "unknown") -> str:
    """Format transcription as SRT subtitle file.

    Args:
        segments: List of dicts with 'text', 'start', 'end' keys
        language: Detected language code

    Returns:
        SRT-formatted subtitle text
    """
    lines = []
    for i, seg in enumerate(segments, 1):
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = format_timestamp_srt(seg.get("start", 0))
        end = format_timestamp_srt(seg.get("end", 0))
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def format_as_json(segments: list[dict], language: str = "unknown") -> str:
    """Format transcription as JSON.

    Args:
        segments: List of dicts with 'text', 'start', 'end' keys
        language: Detected language code

    Returns:
        JSON-formatted transcription
    """
    import json
    output = {
        "language": language,
        "segment_count": len(segments),
        "segments": segments,
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


def format_transcription(
    segments: list[dict],
    format: str = "txt",
    language: str = "unknown",
) -> str:
    """Format transcription in the requested format.

    Args:
        segments: List of dicts with 'text', 'start', 'end' keys
        format: Output format ('txt', 'md', 'srt', 'json')
        language: Detected language code

    Returns:
        Formatted transcription string
    """
    format = format.lower()
    if format == "txt":
        return format_as_txt(segments, language)
    elif format == "md":
        return format_as_md(segments, language)
    elif format == "srt":
        return format_as_srt(segments, language)
    elif format == "json":
        return format_as_json(segments, language)
    else:
        raise ValueError(f"Unsupported format: {format}. Supported: {SUPPORTED_FORMATS}")


def split_text_into_chunks(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    overlap: int = 200,
) -> list[str]:
    """Split text into chunks for AI context limits.

    Splits at sentence boundaries when possible to avoid cutting
    mid-sentence.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk
        overlap: Character overlap between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to find a sentence boundary near the end
        best_cut = end
        for punct in [". ", "! ", "? ", "\n\n", ".", "!", "?", "\n"]:
            idx = text.rfind(punct, start + max_chars // 2, end)
            if idx != -1:
                best_cut = idx + len(punct)
                break

        chunk = text[start:best_cut].strip()
        if chunk:
            chunks.append(chunk)

        start = max(best_cut - overlap, start + 1)

    return chunks


def split_text_by_sentences(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS_PER_CHUNK,
) -> list[str]:
    """Split text into chunks at sentence boundaries.

    More aggressive sentence-boundary splitting than split_text_into_chunks.
    """
    if not text:
        return []

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            if current:
                current += " " + sentence
            else:
                current = sentence
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def assess_transcript_quality(
    segments: list[dict],
    duration_s: float | None = None,
) -> dict:
    """Cheap, deterministic check for a broken transcription.

    Catches the three failures that actually happen on quiet phone recordings:
    nothing recognised, the voice-activity filter eating most of the speech,
    and a decoder stuck looping on one phrase. It does not judge accuracy -
    a merely imperfect transcript passes.

    Args:
        segments: List of dicts with 'text', 'start', 'end' keys
        duration_s: Media duration in seconds; word rate is skipped without it

    Returns:
        Dict with 'ok', 'reasons', 'words', 'wpm' and 'unique_ratio'
    """
    words: list[str] = []
    for seg in segments:
        words.extend((seg.get("text") or "").split())

    reasons: list[str] = []
    if not words:
        return {
            "ok": False,
            "reasons": ["no-speech"],
            "words": 0,
            "wpm": 0.0,
            "unique_ratio": 0.0,
        }

    wpm = 0.0
    if duration_s and float(duration_s) > 0:
        wpm = len(words) / (float(duration_s) / 60.0)
        if float(duration_s) >= 60.0 and wpm < MIN_WORDS_PER_MINUTE:
            reasons.append("low-word-rate")

    lowered = [word.lower().strip(".,!?;:\"'()-") for word in words]
    lowered = [word for word in lowered if word]
    unique_ratio = len(set(lowered)) / len(lowered) if lowered else 0.0
    if len(lowered) >= MIN_WORDS_FOR_RATIO and unique_ratio < MIN_UNIQUE_WORD_RATIO:
        reasons.append("repetition-loop")

    if len(lowered) >= LOOP_MIN_GRAMS * LOOP_GRAM_WORDS:
        grams = [
            " ".join(lowered[i:i + LOOP_GRAM_WORDS])
            for i in range(len(lowered) - LOOP_GRAM_WORDS + 1)
        ]
        if grams:
            top = max(grams.count(gram) for gram in set(grams))
            if top >= max(LOOP_MIN_REPEATS, len(grams) // 20):
                if "repetition-loop" not in reasons:
                    reasons.append("repetition-loop")

    return {
        "ok": not reasons,
        "reasons": reasons,
        "words": len(words),
        "wpm": round(wpm, 1),
        "unique_ratio": round(unique_ratio, 3),
    }


def validate_audio_file(file_path: str) -> tuple[bool, str]:
    """Validate that a file is a supported audio format.

    Args:
        file_path: Path to the audio file

    Returns:
        (is_valid, error_message)
    """
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        return False, f"File not found: {file_path}"

    if not path.is_file():
        return False, f"Not a file: {file_path}"

    valid_extensions = {
        ".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg",
        ".opus", ".webm", ".aac", ".wma",
    }

    ext = path.suffix.lower()
    if ext not in valid_extensions:
        return False, f"Unsupported audio format: {ext}"

    if path.stat().st_size == 0:
        return False, "File is empty"

    return True, ""


def validate_video_file(file_path: str) -> tuple[bool, str]:
    """Validate that a file is a supported video format.

    Args:
        file_path: Path to the video file

    Returns:
        (is_valid, error_message)
    """
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        return False, f"File not found: {file_path}"

    if not path.is_file():
        return False, f"Not a file: {file_path}"

    valid_extensions = {
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv",
        ".wmv", ".mpeg", ".mpg", ".3gp", ".ts",
    }

    ext = path.suffix.lower()
    if ext not in valid_extensions:
        return False, f"Unsupported video format: {ext}"

    if path.stat().st_size == 0:
        return False, "File is empty"

    return True, ""


def get_media_duration_seconds(file_path: str) -> float:
    """Get duration of audio/video file in seconds using ffprobe."""
    from unified.audio_splitter import get_audio_duration
    return get_audio_duration(Path(file_path))


def estimate_transcription_time(duration_seconds: float, model_type: str = "whisper") -> float:
    """Estimate transcription time in seconds.

    Rough estimates based on typical real-time factors:
    - whisper: ~0.3x real-time (30s audio = ~9s processing)
    - sherpa-onnx: ~0.1x real-time
    - nemotron: ~0.5x real-time

    Args:
        duration_seconds: Audio duration in seconds
        model_type: Model type ('whisper', 'sherpa', 'nemotron')

    Returns:
        Estimated processing time in seconds
    """
    factors = {
        "whisper": 0.3,
        "sherpa": 0.1,
        "nemotron": 0.5,
        "gemma": 0.8,
    }
    factor = factors.get(model_type, 0.3)
    return duration_seconds * factor


def merge_transcript_segments(
    segments_list: list[list[dict]],
    separator: str = "\n\n---\n\n",
) -> list[dict]:
    """Merge multiple transcript segment lists into one.

    Adjusts timestamps if segments come from different files
    (assumes they're already in order).

    Args:
        segments_list: List of segment lists from different files
        separator: Separator text between files

    Returns:
        Merged list of segments
    """
    all_segments = []
    time_offset = 0.0

    for segments in segments_list:
        for seg in segments:
            new_seg = dict(seg)
            new_seg["start"] = seg.get("start", 0) + time_offset
            new_seg["end"] = seg.get("end", 0) + time_offset
            all_segments.append(new_seg)

        # Add separator as a text segment
        if segments:
            last_end = segments[-1].get("end", 0)
            time_offset = last_end

    return all_segments