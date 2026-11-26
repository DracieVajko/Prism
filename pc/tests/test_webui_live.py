# pc/tests/test_webui_live.py
from pathlib import Path

HTML = Path(__file__).resolve().parents[1] / "webui_unified.html"

def test_live_controls_present_and_secure_context_guard():
    s = HTML.read_text(encoding="utf-8")
    for ident in ("liveMic", "liveDevice", "btnLiveStart", "btnLiveStop",
                  "liveStatus", "liveTailMic", "liveTailDevice",
                  "liveLinkMic", "liveLinkDevice"):
        assert f'id="{ident}"' in s, ident
    assert "isSecureContext" in s
    assert "getUserMedia" in s and "getDisplayMedia" in s
    for fn in ("startLive", "stopLive", "appendLive"):
        assert f"function {fn}" in s, fn

def test_live_never_uses_media_recorder():
    s = HTML.read_text(encoding="utf-8")
    assert "MediaRecorder" not in s
