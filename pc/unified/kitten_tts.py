"""KittenTTS mini 0.8 — lightweight English TTS (80M params, 8 voices, 24 kHz).

Runs on CPU via ONNX (no PyTorch needed at inference). English only.
Wraps the `kittenml` package; auto-downloads on first use.
"""
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent.parent
MODELS_DIR = HERE / "models"
MODEL_ID = "KittenML/kitten-tts-mini-0.8"

_lock = threading.Lock()
_tts = None
_last_error = ""


def is_installed() -> bool:
    """True when the kittenml package is importable (weights auto-download)."""
    try:
        import kittenml  # noqa: F401
        return True
    except Exception:
        return False


def get_tts():
    global _tts, _last_error
    with _lock:
        if _tts is not None:
            return _tts
        try:
            from kittenml import KittenTTS
            _tts = KittenTTS(MODEL_ID)
            _last_error = ""
            return _tts
        except Exception as e:
            _last_error = f"kitten-tts load failed: {str(e)[:200]}"
            logger.error(_last_error)
            return None


def last_error() -> str:
    return _last_error


def voices() -> list[str]:
    tts = get_tts()
    if tts is None:
        return []
    try:
        return list(tts.available_voices)
    except Exception:
        return []


def synthesize(text: str, voice: str = "Jasper", speed: float = 1.0):
    """Return (samples: list[float], sample_rate: int)."""
    tts = get_tts()
    if tts is None:
        raise RuntimeError(_last_error or "kitten-tts unavailable")
    audio = tts.generate(text, voice=voice)
    return list(audio), 24000


def synthesize_bytes(text: str, voice: str = "Jasper", speed: float = 1.0,
                     fmt: str = "wav") -> bytes:
    samples, sr = synthesize(text, voice, speed)
    import array, io, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        pcm = array.array("h")
        for s in samples:
            pcm.append(int(max(-1.0, min(1.0, s)) * 32767))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def release() -> bool:
    global _tts
    with _lock:
        had = _tts is not None
        _tts = None
        return had
