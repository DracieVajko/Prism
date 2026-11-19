"""Lazy loading: ResidentModelGate idle unload + engine switching."""
import time

from unified.resident_model import ResidentModelGate


def test_touch_updates_last_used():
    mm = ResidentModelGate()
    mm.ensure("audio", "tiny")
    t0 = mm.last_used
    assert t0 > 0
    time.sleep(0.05)
    mm.ensure("audio", "tiny")  # already resident -> must refresh
    assert mm.last_used > t0


def test_idle_seconds():
    mm = ResidentModelGate()
    assert mm.idle_seconds() >= 0
    mm.ensure("image", "A")
    assert mm.idle_seconds() < 2


def test_idle_unload_hook_called():
    calls = []
    mm = ResidentModelGate()
    mm.on_unload = lambda reason: calls.append(reason)
    mm.ensure("image", "A")
    assert mm.unload("idle") is True
    assert calls == ["idle"]


def test_status_reports_idle():
    mm = ResidentModelGate()
    mm.ensure("audio", "small")
    st = mm.status()
    assert st["idle_seconds"] is not None
