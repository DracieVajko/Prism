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

def test_start_claims_session_before_first_await():
    """Regression: double-clicking Start must not open a second orphaned session.

    The LIVE_STARTING claim and the Start-button disable must be taken
    synchronously, before the create POST / permission awaits, and the claim
    must be released on failure so a retry is possible.
    """
    s = HTML.read_text(encoding="utf-8")
    a = s.index("async function startLive()")
    b = s.index("async function stopLive()", a)
    body = s[a:b]

    guard = body[:body.index("const mic")]
    assert "LIVE || LIVE_STARTING" in guard, "guard must block a starting session"
    assert body.count("fetch('/v1/live/sessions'") == 1, "exactly one create POST"

    create_post = body.index("fetch('/v1/live/sessions'")
    before = body[:create_post]
    assert "LIVE_STARTING = true" in before, "claim must be synchronous"
    assert "getElementById('btnLiveStart')" in before, "Start button located"
    assert ".disabled = true" in before, "Start disabled before the create POST"
    assert "LIVE_STARTING = false" in body, "claim released after create"
    assert "setLiveButtons(false)" in body, "buttons restored on fetch failure"
