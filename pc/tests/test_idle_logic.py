"""Idle unload decision: must not unload during a request or below the limit."""
import time

from unified.resident_model import ResidentModelGate
from unified_server import should_idle_unload


def test_no_model_never_unloads():
    mm = ResidentModelGate()
    assert should_idle_unload(mm, 20) is False


def test_fresh_model_does_not_unload():
    mm = ResidentModelGate()
    mm.ensure("image", "A")
    assert should_idle_unload(mm, 20) is False


def test_model_above_limit_unloads():
    mm = ResidentModelGate()
    mm.ensure("image", "A")
    mm.last_used = time.time() - 100
    assert should_idle_unload(mm, 20) is True


def test_in_flight_blocks_unload_even_when_old():
    mm = ResidentModelGate()
    mm.ensure("image", "A")
    mm.last_used = time.time() - 1000
    mm.begin()
    assert should_idle_unload(mm, 20) is False
    mm.end()
    # end() restarts the idle countdown -> not time yet
    assert should_idle_unload(mm, 20) is False
    mm.last_used = time.time() - 1000
    assert should_idle_unload(mm, 20) is True
