"""Deterministic digest helpers in unified.meeting_digest."""
import unified.meeting_digest as md

SEGMENTS = [
    {"start": 0.0, "end": 10.0, "text": "hello there"},
    {"start": 40.0, "end": 50.0, "text": "after a pause"},
    {"start": 50.0, "end": 60.0, "text": "more"},
]


def test_speech_stats_count_speech_not_file_length():
    stats = md.speech_stats(SEGMENTS, duration_s=60.0)
    assert stats["segment_count"] == 3
    assert stats["speech_s"] == 30.0
    assert stats["speech_ratio"] == 0.5
    assert stats["words"] == 6
    assert stats["wpm_speech"] == 12.0


def test_speech_stats_survive_zero_duration():
    stats = md.speech_stats([], 0)
    assert stats["speech_ratio"] == 0.0
    assert stats["wpm_speech"] == 0.0


def test_find_long_pauses_reports_gap_and_trailing_text():
    pauses = md.find_long_pauses(SEGMENTS, min_gap_s=25.0)
    assert len(pauses) == 1
    assert pauses[0]["at_s"] == 10.0
    assert pauses[0]["gap_s"] == 30.0
    assert pauses[0]["after"] == "hello there"


def test_find_long_pauses_ignores_normal_turnover():
    assert md.find_long_pauses(SEGMENTS, min_gap_s=60.0) == []


def test_chunk_segments_keeps_every_line_in_order():
    chunks = md.chunk_segments(SEGMENTS, max_chars=30)
    lines = [line for chunk in chunks for line in chunk["lines"]]
    assert lines == [
        "[00:00:00] hello there",
        "[00:00:40] after a pause",
        "[00:00:50] more",
    ]
    assert [chunk["index"] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk["total"] == len(chunks) for chunk in chunks)
    assert len(chunks) > 1


def test_chunk_segments_splits_between_segments_only():
    chunks = md.chunk_segments(SEGMENTS, max_chars=1000)
    assert len(chunks) == 1
    single = [{"start": 0.0, "end": 5.0, "text": "x" * 500}]
    assert len(md.chunk_segments(single, max_chars=100)) == 1


def test_chunk_segments_rejects_bad_max_chars():
    try:
        md.chunk_segments(SEGMENTS, max_chars=0)
    except ValueError as exc:
        assert "max_chars" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_render_digest_has_stats_pauses_and_chunks():
    text = md.render_digest("20260101_120000", SEGMENTS, duration_s=60.0,
                            language="sk", max_chars=30)
    assert text.startswith("# Recording digest: 20260101_120000")
    assert "Language: sk" in text
    assert "Speech: 0.5 min (50.0% of the file)" in text
    assert "## Long pauses" in text
    assert "[00:00:40] after a pause" in text
    assert "### Chunk 1/" in text
    assert text.endswith("\n")


def test_render_digest_handles_no_speech():
    text = md.render_digest("empty", [], duration_s=0)
    assert "None." in text
    assert "Chunks: 0" in text


def test_digest_index_points_at_the_real_chunk_line():
    many = [
        {"start": float(i), "end": float(i) + 0.9, "text": f"segment number {i} spoken"}
        for i in range(40)
    ]
    text = md.render_digest("long", many, duration_s=40.0, max_chars=200)
    lines = text.splitlines()
    rows = [line for line in lines if line.startswith("- Chunk ")]
    assert len(rows) > 1
    for row in rows:
        label = row.split()[2]
        target = int(row.rsplit("line ", 1)[1])
        assert lines[target - 1].startswith(f"### Chunk {label} ")


def test_quiet_chunk_is_flagged_for_skipping():
    sparse = [
        {"start": 0.0, "end": 1.0, "text": "one"},
        {"start": 120.0, "end": 121.0, "text": "two"},
        {"start": 300.0, "end": 301.0, "text": "three"},
    ]
    chunks = md.chunk_segments(sparse, max_chars=10_000)
    assert len(chunks) == 1
    assert chunks[0]["quiet"] is True
    assert "mostly silence" in md.render_digest("sparse", sparse, 301.0, max_chars=10_000)
