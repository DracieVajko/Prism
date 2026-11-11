"""Sherpa-ONNX streaming ASR as an alternative PC transcription engine.

Models: k2-fsa/sherpa-onnx-streaming-zipformer-*.onnx (or Whisper via Sherpa-ONNX)
Provides causal streaming ASR with ~80-320 ms latency vs whisper.cpp's
500-1500 ms per chunk.

Enabled with PRISM_ASR_ENGINE=sherpa, or per-request with
`model=sherpa-zipformer-small` on /v1/audio/transcriptions.
"""

import io
import logging
import os
import subprocess
import wave
from pathlib import Path
from typing import Optional

from unified.asr_common import model_dir, ram_allows

logger = logging.getLogger("sherpa_asr")

MODEL_ID = "sherpa-zipformer-small"
MAX_SECONDS_PER_CALL = 60.0  # streaming model handles long audio natively

# Model directories under pc/models/
MODEL_DIRS = {
    "sherpa-zipformer-small": "sherpa-zipformer-small",
    "sherpa-zipformer-medium": "sherpa-zipformer-medium",
    "sherpa-zipformer-large": "sherpa-zipformer-large",
}

# Rough resident memory requirement per variant
MODEL_MIN_RAM_GB = {
    "sherpa-zipformer-small": 0.5,
    "sherpa-zipformer-medium": 1.0,
    "sherpa-zipformer-large": 2.0,
}

_active_id = MODEL_ID


def _model_dir(model_id: str = None) -> Path:
    mid = model_id or _active_id
    return model_dir(MODEL_DIRS.get(mid, mid), MODEL_ID)


def is_installed(model_id: str = None) -> bool:
    d = _model_dir(model_id)
    # Sherpa-ONNX models are typically .onnx files + tokens.txt
    has_encoder = any(d.glob("*encoder*.onnx")) if d.exists() else False
    has_decoder = any(d.glob("*decoder*.onnx")) if d.exists() else False
    has_joiner = any(d.glob("*joiner*.onnx")) if d.exists() else False
    has_tokens = (d / "tokens.txt").exists() if d.exists() else False
    return has_encoder and has_decoder and has_joiner and has_tokens


def enabled() -> bool:
    return os.environ.get("PRISM_ASR_ENGINE", "").strip().lower() == "sherpa"


_processor = None
_model = None
_recognizer = None
_device = None
_active_id = MODEL_ID


def init(model_id: str = None) -> bool:
    """Load a Sherpa-ONNX ASR model. Returns False and logs the reason on failure."""
    global _recognizer, _active_id
    mid = model_id or _active_id
    if _recognizer is not None and mid == _active_id:
        return True
    if _recognizer is not None and mid != _active_id:
        release()
    _active_id = mid
    if not is_installed(mid):
        logger.warning("Sherpa-ONNX ASR not installed at %s", _model_dir(mid))
        return False
    needed = MODEL_MIN_RAM_GB.get(mid, 0.5)
    fits, have = ram_allows(needed)
    if not fits:
        logger.error("Sherpa-ONNX %s needs ~%.0f GB RAM, this machine has %.1f GB",
                     mid, needed, have)
        return False
    try:
        import sherpa_onnx
        d = str(_model_dir(mid))
        # Find model files
        encoder = None
        decoder = None
        joiner = None
        for f in Path(d).glob("*.onnx"):
            name = f.name.lower()
            if "encoder" in name and encoder is None:
                encoder = str(f)
            elif "decoder" in name and decoder is None:
                decoder = str(f)
            elif "joiner" in name and joiner is None:
                joiner = str(f)
        tokens = str(d / "tokens.txt")
        if not all([encoder, decoder, joiner]):
            logger.warning("Sherpa-ONNX model files incomplete in %s", d)
            return False
        config = sherpa_onnx.OnlineRecognizerConfig(
            model_config=sherpa_onnx.OnlineRecognizerConfig.ModelConfig(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                num_threads=4,
                decode_chunk_size=16,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.0,
                rule2_min_trailing_silence=0.0,
                rule3_min_utterance_length=200,
            ),
            decode_config=sherpa_onnx.OnlineRecognizerConfig.DecodeConfig(
                decoding_method="greedy_search",
                max_active_paths=4,
                hotwords_score=1.5,
                hotwords_file="",
            ),
            endpoint_config=sherpa_onnx.OnlineRecognizerConfig.EndpointConfig(
                rule1_min_trailing_silence=2.0,
                rule2_min_trailing_silence=0.0,
                rule3_min_utterance_length=200,
            ),
        )
        _recognizer = sherpa_onnx.OnlineRecognizer(config)
        logger.info("Sherpa-ONNX ASR %s loaded", mid)
        return True
    except Exception as e:
        logger.error("Sherpa-ONNX init failed: %s", e)
        _recognizer = None
        return False


def release():
    global _recognizer
    _recognizer = None


def _normalise_language(code: Optional[str] = None) -> str:
    """Map a language code to Sherpa-ONNX supported language."""
    if not code:
        return "auto"
    c = code.strip().lower().replace("_", "-")
    # Sherpa-ONNX models typically support a limited set of languages
    known = {
        "en", "zh", "yue", "ja", "ko", "de", "fr", "es", "pt", "it", "ru",
        "nl", "uk", "ar", "hi", "th", "vi", "tr", "pl", "nl", "id", "ms",
    }
    base = c.split("-")[0]
    if base in known:
        return base
    return "en"  # Sherpa-ONNX defaults to English


def _read_audio_16k(path: str):
    """Return (float32 samples, sample_rate) for any container ffmpeg reads.

    Sherpa-ONNX only consumes raw samples, so mp3/mp4 uploads are normalised
    through ffmpeg instead of being mistaken for raw PCM.
    """
    try:
        with wave.open(path, "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        logger.debug("wave.open failed for %s (%s); decoding with ffmpeg", path, exc)
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000",
             "-f", "wav", "pipe:1"],
            capture_output=True, check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            detail = (proc.stderr or b"").decode("utf-8", "replace")[-300:]
            raise RuntimeError(f"could not decode audio: {detail}") from exc
        with wave.open(io.BytesIO(proc.stdout), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())

    import numpy as np
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def transcribe(audio_path: str, language: Optional[str] = None,
               model_id: str = None) -> tuple[str, str]:
    """Transcribe one audio file using Sherpa-ONNX streaming.

    Returns (text, language_used).
    """
    if model_id and model_id != _active_id:
        if not init(model_id):
            raise RuntimeError(f"Sherpa-ONNX model {model_id} not available")
    elif _recognizer is None and not init():
        raise RuntimeError("Sherpa-ONNX ASR not available")

    lang = _normalise_language(language)
    samples, sample_rate = _read_audio_16k(str(audio_path))

    # Stream through Sherpa-ONNX recognizer
    stream = _recognizer.create_stream()
    chunk_size = int(sample_rate * 0.1)  # 100 ms chunks
    for i in range(0, len(samples), chunk_size):
        stream.accept_waveform(sample_rate, samples[i:i + chunk_size])
        while stream.is_ready():
            stream.decode()

    # ``stream.result.text`` is cumulative, so read it once after the audio is
    # finished instead of appending the whole transcript after every chunk.
    stream.input_finished()
    while stream.is_ready():
        stream.decode()
    return (stream.result.text or "").strip(), lang or "en"
