"""Sherpa-ONNX streaming ASR as an alternative PC transcription engine.

Models: k2-fsa/sherpa-onnx-streaming-zipformer-*.onnx (or Whisper via Sherpa-ONNX)
Provides causal streaming ASR with ~80-320 ms latency vs whisper.cpp's
500-1500 ms per chunk.

Enabled with PRISM_ASR_ENGINE=sherpa, or per-request with
`model=sherpa-zipformer-small` on /v1/audio/transcriptions.
"""

import logging
import os
from pathlib import Path
from typing import Optional

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
    sub = MODEL_DIRS.get(mid, mid)
    return Path(__file__).resolve().parent.parent / "models" / sub


def is_installed(model_id: str = None) -> bool:
    d = _model_dir(model_id)
    # Sherpa-ONNX models are typically .onnx files + tokens.txt
    has_encoder = any(d.glob("*encoder*.onnx")) if d.exists() else False
    has_decoder = any(d.glob("*decoder*.onnx")) if d.exists() else False
    has_joiner = any(d.glob("*joiner*.onnx")) if d.exists() else False
    has_tokens = (d / "tokens.txt").exists() if d.exists() else False
    return has_encoder and has_decoder and has_joiner and has_tokens


def _total_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return 0.0


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
    have = _total_ram_gb()
    if have and have + 0.5 < needed:
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

    import sherpa_onnx
    import wave

    lang = _normalise_language(language)

    # Read audio file
    try:
        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
    except Exception:
        # Fallback: try reading as raw PCM
        audio_data = Path(audio_path).read_bytes()
        sample_rate = 16000

    # Convert to float32 array
    import numpy as np
    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Stream through Sherpa-ONNX recognizer
    stream = _recognizer.create_stream()
    chunk_size = int(sample_rate * 0.1)  # 100 ms chunks
    parts = []
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        stream.accept_waveform(sample_rate, chunk)
        while stream.is_ready():
            stream.decode()
        text = stream.result.text
        if text:
            parts.append(text)

    # Flush remaining
    stream.input_finished()
    while stream.is_ready():
        stream.decode()
    text = stream.result.text
    if text:
        parts.append(text)

    final_text = " ".join(parts).strip()
    return final_text, lang or "en"
