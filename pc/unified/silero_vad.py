"""Silero VAD (Voice Activity Detection) for Prism PC server.

Silero VAD is a tiny (~2 MB) ONNX model that detects speech segments in audio.
Used as a pre-filter before transcription to avoid processing silence, which
saves time and improves transcript quality.

Model: silero-vad/v5.onnx (ONNX Runtime compatible)
"""

import logging
import os
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger("silero_vad")

MODEL_ID = "silero-vad"
MODEL_DIR_NAME = "silero-vad"

_model = None
_utils = None


def model_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / MODEL_DIR_NAME


def is_installed() -> bool:
    d = model_dir()
    onnx_files = list(d.glob("*.onnx")) if d.exists() else []
    return len(onnx_files) > 0


def _total_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return 0.0


def init() -> bool:
    """Load Silero VAD model. Returns False if not available."""
    global _model, _utils
    if _model is not None:
        return True
    if not is_installed():
        logger.warning("Silero VAD not installed at %s", model_dir())
        return False
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir=str(model_dir()),
            model="silero_vad",
            source="local",
            trust_repo=True,
        )
        _model = model
        _utils = utils
        logger.info("Silero VAD loaded from %s", model_dir())
        return True
    except Exception as e:
        logger.error("Silero VAD init failed: %s", e)
        return False


def release():
    global _model, _utils
    _model = None
    _utils = None


def detect_speech_segments(
    audio_path: str,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 512,
) -> List[Tuple[float, float]]:
    """Detect speech segments in an audio file.

    Returns list of (start_seconds, end_seconds) tuples for speech regions.
    """
    if _model is None:
        if not init():
            raise RuntimeError("Silero VAD not available")

    import torch
    from torchaudio import load as load_audio

    wav, sr = load_audio(audio_path)
    if sr != sample_rate:
        import torchaudio.transforms as T
        resampler = T.Resample(sr, sample_rate)
        wav = resampler(wav)

    # Get speech timestamps
    from silero_vad import get_speech_timestamps
    timestamps = get_speech_timestamps(
        wav,
        _model,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        window_size_samples=window_size_samples,
        sampling_rate=sample_rate,
        return_seconds=True,
    )

    return [(t["start"], t["end"]) for t in timestamps]


def transcribe_with_vad(
    audio_path: str,
    language: Optional[str] = None,
    transcribe_fn=None,
    sample_rate: int = 16000,
) -> tuple[str, str, List[Tuple[float, float]]]:
    """Transcribe audio using VAD pre-filtering.

    Only speech segments are sent to the transcription engine, saving time
    on silence and improving transcript quality.

    Returns (text, language, segments) where segments is a list of
    (start_sec, end_sec) tuples for detected speech regions.
    """
    if transcribe_fn is None:
        raise ValueError("transcribe_fn is required")

    speech_segments = detect_speech_segments(audio_path, sample_rate=sample_rate)

    if not speech_segments:
        return "", "en", []

    # Extract speech segments and transcribe each
    import torch
    from torchaudio import load as load_audio
    import tempfile
    import os

    wav, sr = load_audio(audio_path)
    if sr != sample_rate:
        import torchaudio.transforms as T
        resampler = T.Resample(sr, sample_rate)
        wav = resampler(wav)

    all_text_parts = []
    for start_sec, end_sec in speech_segments:
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        segment_wav = wav[:, start_sample:end_sample]

        # Save segment to temp file
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, f"segment_{start_sec}_{end_sec}.wav")
        import torchaudio
        torchaudio.save(tmp_path, segment_wav, sample_rate)

        try:
            text, lang = transcribe_fn(tmp_path, language)
            if text:
                all_text_parts.append(text)
        finally:
            try:
                os.remove(tmp_path)
                os.rmdir(tmp_dir)
            except Exception:
                pass

    return " ".join(all_text_parts), language or "en", speech_segments
