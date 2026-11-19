"""OpenAI transcription WebSocket: Cindy's event sequence, no model load."""
import base64

from fastapi.testclient import TestClient

import unified_server as us


def test_realtime_session_flow(monkeypatch):
    monkeypatch.setattr(
        us, "_transcribe_pcm",
        lambda raw, rate, wanted, language: ("hello world", "en"))
    monkeypatch.setattr(us._manager, "ensure", lambda kind, model_id: model_id)

    client = TestClient(us.app)
    with client.websocket_connect("/v1/realtime") as ws:
        ws.send_json({
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "turbo", "language": "en"},
                    "turn_detection": None,
                }},
            },
        })
        assert ws.receive_json()["type"] == "session.updated"

        pcm = (b"\x01\x00" * 4800)  # 200 ms of 24 kHz PCM16 mono
        ws.send_json({"type": "input_audio_buffer.append",
                      "audio": base64.b64encode(pcm).decode()})
        ws.send_json({"type": "input_audio_buffer.commit"})

        committed = ws.receive_json()
        assert committed["type"] == "input_audio_buffer.committed"
        item_id = committed["item_id"]

        done = ws.receive_json()
        assert done["type"].endswith("completed")
        assert done["item_id"] == item_id
        assert done["transcript"] == "hello world"

        ws.send_json({"type": "session.finish"})
        assert ws.receive_json()["type"] == "session.finished"
