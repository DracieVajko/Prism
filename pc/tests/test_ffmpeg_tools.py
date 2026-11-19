"""FFmpeg preset metadata; no ffmpeg process is started."""
import unified.ffmpeg_tools as ft


def test_presets_have_all_families():
    presets = ft.get_presets()
    assert set(presets) == {"video", "audio", "image", "resize"}
    assert presets["video"]
    assert presets["audio"]


def test_missing_input_returns_empty_info(tmp_path):
    info = ft.get_media_info(tmp_path / "missing.mp4")
    assert isinstance(info, dict)
    assert not info
