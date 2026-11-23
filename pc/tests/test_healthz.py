"""Cindy liveness probe contract: /healthz must answer in every model state.

Regression guard: `whisper_name` was only assigned on the "audio is resident"
branch, so a cold (or idle-unloaded) server raised UnboundLocalError -> 500.
"""
from unified_server import healthz


def test_healthz_answers_with_nothing_loaded():
    probe = healthz()
    assert probe["ok"] is True
    assert probe["loaded"] is False
    assert probe["loaded_model"] is None
    # configured model, never None, so a caller can always compare it
    assert isinstance(probe["asr_model"], str) and probe["asr_model"]
    assert isinstance(probe["audio_whisper_models_resident"], int)
    assert probe["uptime_s"] >= 0


def test_healthz_reports_loaded_model_when_resident():
    import unified_server as us

    us._manager._loaded_kind = "audio"
    us._manager._loaded_id = "turbo"
    try:
        probe = healthz()
        assert probe["loaded"] is True
        assert probe["loaded_model"] == "turbo"
        assert probe["asr_model"] == "turbo"
    finally:
        us._manager._loaded_kind = None
        us._manager._loaded_id = None
