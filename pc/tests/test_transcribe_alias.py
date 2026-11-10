"""The /transcribe alias must call the real endpoint with valid kwargs."""
import asyncio
import inspect

import unified_server as us


def test_transcribe_alias_forwards_only_known_kwargs(monkeypatch):
    real = us.transcriptions
    captured = {}

    async def fake(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(us, "transcriptions", fake)
    asyncio.run(us.transcribe_alias(file=None, language="en"))

    # bind() raises TypeError when the alias passes a parameter the real
    # endpoint does not accept (the previous `model_name` bug).
    inspect.signature(real).bind(**captured)
    assert "model" in captured
    assert captured["language"] == "en"
