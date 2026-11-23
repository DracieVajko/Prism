"""
Deterministic digest of a long recording.

A one-hour meeting transcript does not fit into a single agent context window,
and plain text loses the timestamps that make a recording navigable. This
module turns ASR segments into a compact Markdown digest that holds:

- speech statistics that need no speaker diarization (duration, speech time,
  word count, pace while speaking),
- long pauses, which usually mark a topic switch or a change of speaker,
- an index of timestamped chunks with the line number where each one starts,
  so a reader can open a bounded range of the file instead of all of it,
- the transcript itself in those chunks.

Nothing here interprets the conversation. It only reports what is measurable
from the segments, so the reading agent keeps every judgement about content,
decisions and action items.
"""

from __future__ import annotations

from unified.transcription_utils import (
    DEFAULT_MAX_CHARS_PER_CHUNK,
    format_timestamp_hms,
)

# A gap at least this long is reported as a pause.
LONG_PAUSE_S = 25.0

# Pauses are an index, not the substance; keep the list readable.
MAX_RENDERED_PAUSES = 20


# A chunk with less speech than this is marked as mostly silence, so a reader
# can skip it.
QUIET_CHUNK_RATIO = 0.3


def _clean(text: str) -> str:
    """Collapse whitespace so one segment renders as one line."""
    return " ".join((text or "").split())


def _start(segment: dict) -> float:
    return float(segment.get("start", 0) or 0)


def _end(segment: dict) -> float:
    return float(segment.get("end", 0) or 0)


def speech_stats(segments: list[dict], duration_s: float) -> dict:
    """Measurable speech statistics. No diarization, so no talk ratio."""
    duration_s = max(0.0, float(duration_s or 0.0))
    speech_s = sum(max(0.0, _end(seg) - _start(seg)) for seg in segments)
    words = sum(len(_clean(seg.get("text", "")).split()) for seg in segments)
    return {
        "segment_count": len(segments),
        "duration_s": round(duration_s, 1),
        "speech_s": round(speech_s, 1),
        "speech_ratio": round(speech_s / duration_s, 3) if duration_s else 0.0,
        "words": words,
        "wpm_speech": round(words / (speech_s / 60.0), 1) if speech_s else 0.0,
    }


def find_long_pauses(
    segments: list[dict],
    min_gap_s: float = LONG_PAUSE_S,
) -> list[dict]:
    """Gaps between segments of at least ``min_gap_s`` seconds.

    Each entry keeps the window it follows, so the reader can find the
    topic that ended there without reopening the whole transcript.
    """
    ordered = sorted(segments, key=_start)
    pauses: list[dict] = []
    for previous, following in zip(ordered, ordered[1:]):
        gap = _start(following) - _end(previous)
        if gap >= min_gap_s:
            pauses.append(
                {
                    "at_s": round(_end(previous), 1),
                    "gap_s": round(gap, 1),
                    "after": _clean(previous.get("text", ""))[-80:],
                }
            )
    return pauses


def chunk_segments(
    segments: list[dict],
    max_chars: int = DEFAULT_MAX_CHARS_PER_CHUNK,
) -> list[dict]:
    """Group timestamped transcript lines into chunks of at most ``max_chars``.

    A single segment longer than ``max_chars`` still becomes its own chunk:
    splitting inside a segment would invent timestamps.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    chunks: list[dict] = []
    current: dict | None = None

    for segment in sorted(segments, key=_start):
        text = _clean(segment.get("text", ""))
        if not text:
            continue
        line = f"[{format_timestamp_hms(_start(segment))}] {text}"

        if current is not None and current["chars"] + len(line) + 1 > max_chars:
            chunks.append(current)
            current = None

        if current is None:
            current = {
                "start_s": round(_start(segment), 1),
                "end_s": round(_end(segment), 1),
                "chars": 0,
                "speech_s": 0.0,
                "words": 0,
                "lines": [],
            }

        current["lines"].append(line)
        current["chars"] += len(line) + 1
        current["speech_s"] += max(0.0, _end(segment) - _start(segment))
        current["words"] += len(text.split())
        current["end_s"] = round(_end(segment), 1)

    if current is not None:
        chunks.append(current)

    total = len(chunks)
    for index, chunk in enumerate(chunks, 1):
        span = max(0.1, chunk["end_s"] - chunk["start_s"])
        chunk["index"] = index
        chunk["total"] = total
        chunk["speech_ratio"] = round(min(1.0, chunk["speech_s"] / span), 3)
        chunk["speech_s"] = round(chunk["speech_s"], 1)
        chunk["quiet"] = chunk["speech_ratio"] < QUIET_CHUNK_RATIO
    return chunks


def render_digest(
    name: str,
    segments: list[dict],
    duration_s: float,
    language: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    long_pause_s: float = LONG_PAUSE_S,
) -> str:
    """Render the Markdown digest for one recording."""
    stats = speech_stats(segments, duration_s)
    pauses = find_long_pauses(segments, long_pause_s)
    chunks = chunk_segments(segments, max_chars)

    lines = [f"# Recording digest: {name}", ""]
    lines.append(f"- Duration: {stats['duration_s'] / 60:.1f} min")
    if language:
        lines.append(f"- Language: {language}")
    lines.append(f"- Segments: {stats['segment_count']}")
    lines.append(
        f"- Speech: {stats['speech_s'] / 60:.1f} min "
        f"({stats['speech_ratio'] * 100:.1f}% of the file)"
    )
    lines.append(f"- Words: {stats['words']}")
    lines.append(
        f"- Pace while speaking: {stats['wpm_speech']:.0f} wpm "
        "(single mixed channel, so this is not a per-speaker metric)"
    )
    lines.append(f"- Chunks: {len(chunks)} of at most {max_chars} characters")
    lines.append("")

    lines.append("## Chunk index")
    lines.append("")
    index_slot = len(lines)
    # One row per chunk keeps the line count fixed, so the transcript below can
    # be laid out once and every row can point at the real line number.
    lines.extend([""] * len(chunks))
    if not chunks:
        lines.append("No speech in the recording.")
    lines.append("")

    lines.append(f"## Long pauses (>= {long_pause_s:.0f} s, possible topic switches)")
    if not pauses:
        lines.append("")
        lines.append("None.")
    else:
        lines.append("")
        for pause in pauses[:MAX_RENDERED_PAUSES]:
            after = pause["after"].replace("|", "/")
            lines.append(
                f"- {format_timestamp_hms(pause['at_s'])} "
                f"({pause['gap_s']:.0f} s) after: \"{after}\""
            )
        hidden = len(pauses) - MAX_RENDERED_PAUSES
        if hidden > 0:
            lines.append(f"- ... and {hidden} more")
    lines.append("")

    lines.append("## Transcript chunks")
    lines.append("")
    starts: list[int] = []
    for chunk in chunks:
        starts.append(len(lines) + 1)
        lines.append(
            f"### Chunk {chunk['index']}/{chunk['total']} "
            f"{format_timestamp_hms(chunk['start_s'])} - "
            f"{format_timestamp_hms(chunk['end_s'])} "
            f"({chunk['chars']} chars)"
        )
        lines.append("")
        lines.extend(chunk["lines"])
        lines.append("")

    for offset, chunk in enumerate(chunks):
        row = (
            f"- Chunk {chunk['index']}/{chunk['total']} "
            f"{format_timestamp_hms(chunk['start_s'])} - "
            f"{format_timestamp_hms(chunk['end_s'])} - "
            f"{chunk['words']} words, {chunk['chars']} chars, "
            f"speech {chunk['speech_ratio'] * 100:.0f}%"
            f"{' (mostly silence)' if chunk['quiet'] else ''} "
            f"- line {starts[offset]}"
        )
        lines[index_slot + offset] = row

    return "\n".join(lines).rstrip() + "\n"
