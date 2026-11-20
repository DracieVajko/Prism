"""On-device-ish TTS for the PC server: Supertonic 3 (int8) via sherpa-onnx.

The model is downloaded lazily on first use (tar.bz2 from the sherpa-onnx
release assets) and extracted under pc/models/supertonic-3-tts-int8. Synthesis
returns PCM; encoding to mp3/aac/opus/... is done with ffmpeg.
"""
import array
import os
import subprocess
import tarfile
import tempfile
import threading
import urllib.request
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
MODELS_DIR = HERE / "models"
MODEL_DIR = MODELS_DIR / "supertonic-3-tts-int8"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2"
)
INNER_DIR = "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"

_lock = threading.Lock()
_tts = None
_last_error = ""


def is_installed() -> bool:
    return (MODEL_DIR / "tts.json").is_file() and (MODEL_DIR / "voice.bin").is_file()


def model_dir() -> Path:
    return MODEL_DIR


def _download_and_extract():
    global _last_error
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = MODELS_DIR / "supertonic-3-tts-int8.tar.bz2.part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tar_path)
        final = MODELS_DIR / "supertonic-3-tts-int8.tar.bz2"
        tar_path.replace(final)
        with tarfile.open(final, "r:bz2") as tf:
            tf.extractall(MODELS_DIR)
        src = MODELS_DIR / INNER_DIR
        if src.is_dir() and not MODEL_DIR.exists():
            src.rename(MODEL_DIR)
        try:
            final.unlink(missing_ok=True)
        except Exception:
            pass
    except Exception as e:
        _last_error = str(e)[:300]
        raise


def get_tts():
    global _tts, _last_error
    if _tts is not None:
        return _tts
    with _lock:
        if _tts is not None:
            return _tts
        if not is_installed():
            _download_and_extract()
        try:
            import sherpa_onnx
        except Exception as e:
            _last_error = f"sherpa_onnx not installed: {e}"
            raise
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                    duration_predictor=str(MODEL_DIR / "duration_predictor.int8.onnx"),
                    text_encoder=str(MODEL_DIR / "text_encoder.int8.onnx"),
                    vector_estimator=str(MODEL_DIR / "vector_estimator.int8.onnx"),
                    vocoder=str(MODEL_DIR / "vocoder.int8.onnx"),
                    tts_json=str(MODEL_DIR / "tts.json"),
                    unicode_indexer=str(MODEL_DIR / "unicode_indexer.bin"),
                    voice_style=str(MODEL_DIR / "voice.bin"),
                ),
                num_threads=max(1, min(4, os.cpu_count() or 2)),
            )
        )
        _tts = sherpa_onnx.OfflineTts(cfg)
        return _tts


def _split_text(text: str, max_chars: int = 260):
    """Split into speakable chunks without breaking sentences mid-word."""
    import re
    text = (text or "").strip()
    if not text:
        return []
    # First try sentence boundaries.
    parts = re.split(r"(?<=[.!?…])\s+", text)
    chunks = []
    cur = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + " " + p).strip() if cur else p
        else:
            if cur:
                chunks.append(cur)
            # Hard-split very long sentences.
            if len(p) > max_chars:
                words = p.split()
                cur = ""
                for w in words:
                    if len(cur) + len(w) + 1 <= max_chars:
                        cur = (cur + " " + w).strip() if cur else w
                    else:
                        if cur:
                            chunks.append(cur)
                        cur = w
                if cur:
                    chunks.append(cur)
                    cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks


def synthesize(text: str, voice: str = "0", speed: float = 1.0):
    """Return (pcm_samples_list, sample_rate)."""
    tts = get_tts()
    try:
        sid = int(str(voice).strip())
    except ValueError:
        sid = 0
    sid = max(0, min(9, sid))
    speed = float(speed or 1.0)
    if speed <= 0:
        speed = 1.0
    chunks = _split_text(text)
    if not chunks:
        chunks = [text.strip() or " "]
    samples = []
    sr = None
    for c in chunks:
        audio = tts.generate(c, sid=sid, speed=speed)
        if sr is None:
            sr = audio.sample_rate
        samples.extend(list(audio.samples))
        # small pause between chunks
        samples.extend([0.0] * int(sr * 0.12))
    if sr is None:
        sr = 24000
    return samples, sr


def _write_wav(samples, sr, path: Path):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(array.array("h", [int(x * 32767) for x in samples]).tobytes())


def synthesize_bytes(text: str, voice: str = "0", speed: float = 1.0,
                     fmt: str = "mp3"):
    """Return (audio_bytes, content_type)."""
    fmt = (fmt or "mp3").lower().strip().lstrip(".")
    if fmt not in ("mp3", "aac", "m4a", "wav", "opus", "ogg", "flac"):
        fmt = "mp3"
    samples, sr = synthesize(text, voice=voice, speed=speed)
    tmpdir = Path(tempfile.mkdtemp(prefix="prism_tts_"))
    wav_path = tmpdir / "out.wav"
    _write_wav(samples, sr, wav_path)
    if fmt == "wav":
        data = wav_path.read_bytes()
        try:
            wav_path.unlink(missing_ok=True)
            tmpdir.rmdir()
        except Exception:
            pass
        return data, "audio/wav"
    out_path = tmpdir / f"out.{fmt}"
    # ffmpeg codec selection
    codec_args = []
    if fmt == "mp3":
        codec_args = ["-c:a", "libmp3lame", "-q:a", "2"]
    elif fmt in ("aac", "m4a"):
        codec_args = ["-c:a", "aac", "-b:a", "128k"]
    elif fmt == "opus":
        codec_args = ["-c:a", "libopus", "-b:a", "96k"]
    elif fmt == "ogg":
        codec_args = ["-c:a", "libvorbis", "-q:a", "4"]
    elif fmt == "flac":
        codec_args = ["-c:a", "flac"]
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(wav_path)] + codec_args + [str(out_path)]
    subprocess.run(cmd, check=True)
    data = out_path.read_bytes()
    for p in (wav_path, out_path):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    try:
        tmpdir.rmdir()
    except Exception:
        pass
    content_types = {
        "mp3": "audio/mpeg", "aac": "audio/aac", "m4a": "audio/mp4",
        "opus": "audio/opus", "ogg": "audio/ogg", "flac": "audio/flac",
    }
    return data, content_types.get(fmt, "application/octet-stream")


def last_error() -> str:
    return _last_error


def voices():
    return [{"id": str(i), "name": f"Voice {i}"} for i in range(10)]
