"""Audio splitter helpers that do not invoke ffmpeg."""
import unified.audio_splitter as aud


def test_format_timestamp():
    assert aud.format_timestamp(0) == "00:00:00"
    assert aud.format_timestamp(3661) == "01:01:01"
    assert aud.format_timestamp(59.9) == "00:00:59"


def test_cleanup_chunks_removes_files(tmp_path):
    first = tmp_path / "chunk_a.wav"
    first.write_bytes(b"x")
    aud.cleanup_chunks([(first, 0.0, 1.0)])
    assert not first.exists()
