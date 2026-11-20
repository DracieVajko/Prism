"""Transcribe every video/audio under the AI Income Challenge folder.

Model: faster-whisper large-v3 (most accurate installed), CUDA float16 with a
CPU int8 fallback. Audio is decoded with ffmpeg to 16 kHz mono and passed to
faster-whisper as a numpy array (its PyAV path is broken in this environment).

Input level handling:
    Recordings made with a very low mic gain land near -50 dB and Whisper's
    VAD then treats the whole file as silence. Unless PRISM_ASR_GAIN_DB is set
    explicitly, the peak level is measured and a gain that brings it to about
    -2 dB is applied automatically. If a transcription still comes back empty,
    it is retried once with a strong gain before being reported as EMPTY.

Outputs per input file, in <root>/_transcripts:
    <day>__<name>.txt    plain text
    <day>__<name>.srt    subtitles
    <day>__<name>.json   metadata + segments

Usage:
    python pc/scripts/transcribe_income.py            # all media
    python pc/scripts/transcribe_income.py VID-2026   # only matching names

Re-running skips files whose .txt already exists and is non-empty.
"""
import json
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Projekty\AI Income Challenge")
OUT_DIR = ROOT / "_transcripts"
MODEL_DIR = r"D:\AI\Prism\pc\models\faster-whisper-large"
FFMPEG = "ffmpeg"
TMP_DIR = Path(os.environ.get("TEMP", ".")) / "prism_asr"
EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v",
        ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}

# A peak below this is treated as "recorded too quietly" and gets auto-gain.
LOW_PEAK_DB = -20.0
TARGET_PEAK_DB = -2.0
MAX_AUTO_GAIN_DB = 50.0

# Whisper options. VAD is on by default; music/very short clips sometimes need
# PRISM_ASR_VAD=0 and PRISM_ASR_BEAM=5 to avoid a one-word VAD artifact.
VAD_FILTER = os.environ.get("PRISM_ASR_VAD", "1").strip().lower() not in (
    "0", "false", "no", "off")
BEAM_SIZE = int(os.environ.get("PRISM_ASR_BEAM", "1"))


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "run.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def add_cuda_dlls():
    base = Path(os.environ.get("APPDATA", "")) / "Python" / "Python314" / "site-packages" / "nvidia"
    added = []
    for parts in (("cublas", "bin"), ("cudnn", "bin"), ("cuda_nvrtc", "bin")):
        d = base.joinpath(*parts)
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + ";" + os.environ.get("PATH", "")
            added.append(str(d))
    return added


def measure_peak_db(path: Path) -> float:
    """Peak sample level of the first audio stream, in dBFS."""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-nostats", "-i", str(path), "-vn",
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, errors="replace",
    )
    match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", proc.stderr or "")
    return float(match.group(1)) if match else 0.0


def resolve_gain(path: Path) -> tuple[float, str]:
    """Return (gain_db, reason). Explicit env wins over auto-detection."""
    env = os.environ.get("PRISM_ASR_GAIN_DB", "").strip()
    if env:
        try:
            return float(env), "env"
        except ValueError:
            log(f"  ignoring invalid PRISM_ASR_GAIN_DB={env!r}")
    peak = measure_peak_db(path)
    if peak < LOW_PEAK_DB:
        gain = round(min(MAX_AUTO_GAIN_DB, TARGET_PEAK_DB - peak), 1)
        return gain, f"auto, peak {peak:.1f} dB"
    return 0.0, f"none, peak {peak:.1f} dB"


def srt_ts(t):
    ms = round(t * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def decode(path: Path, gain_db: float) -> tuple[np.ndarray, float]:
    wav_path = TMP_DIR / (path.stem.replace(" ", "_") + ".wav")
    cmd = [FFMPEG, "-y", "-v", "error", "-i", str(path), "-vn",
           "-ac", "1", "-ar", "16000"]
    if gain_db:
        cmd += ["-af", f"volume={gain_db}dB"]
    cmd.append(str(wav_path))
    subprocess.run(cmd, check=True)
    with wave.open(str(wav_path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            raise RuntimeError(
                f"unexpected wav format: {w.getframerate()} Hz, {w.getnchannels()} ch")
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    wav_path.unlink(missing_ok=True)
    audio = audio.astype(np.float32) / 32768.0
    return audio, len(audio) / 16000.0


def target_stem(path: Path) -> str:
    rel = path.relative_to(ROOT)
    parts = [p.replace(" ", "_") for p in rel.parts]
    return "__".join(parts[:-1] + [rel.stem.replace(" ", "_")])


def load_model(device, compute):
    from faster_whisper import WhisperModel
    return WhisperModel(MODEL_DIR, device=device, compute_type=compute)


def run_transcription(model, audio):
    segments_iter, info = model.transcribe(
        audio, beam_size=BEAM_SIZE, vad_filter=VAD_FILTER,
        condition_on_previous_text=False)
    segments = [
        {"start": round(float(s.start), 2), "end": round(float(s.end), 2),
         "text": (s.text or "").strip()}
        for s in segments_iter if (s.text or "").strip()
    ]
    return segments, info


def write_outputs(stem, path, duration, segments, info, gain_db, gain_reason):
    text = " ".join(s["text"] for s in segments).strip()
    tmp_txt = OUT_DIR / f"{stem}.txt.tmp"
    tmp_txt.write_text(text, encoding="utf-8")
    tmp_txt.replace(OUT_DIR / f"{stem}.txt")

    srt = "\n".join(
        f"{i}\n{srt_ts(s['start'])} --> {srt_ts(s['end'])}\n{s['text']}\n"
        for i, s in enumerate(segments, 1)
    )
    (OUT_DIR / f"{stem}.srt").write_text(srt, encoding="utf-8")
    (OUT_DIR / f"{stem}.json").write_text(json.dumps({
        "source": str(path),
        "model": "faster-whisper-large",
        "language": getattr(info, "language", None),
        "duration_s": round(duration, 1),
        "gain_db": gain_db,
        "gain_source": gain_reason,
        "segments": segments,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return text


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    log(f"cuda dll dirs: {add_cuda_dlls()}")

    files = sorted(
        p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in EXTS
    )
    if len(sys.argv) > 1:
        needle = sys.argv[1].lower()
        files = [p for p in files if needle in str(p).lower()]
    log(f"{len(files)} media files, model=large-v3")

    model = None
    device = "cuda"
    for path in files:
        stem = target_stem(path)
        out_txt = OUT_DIR / f"{stem}.txt"
        if out_txt.exists() and out_txt.stat().st_size > 0:
            log(f"skip (done): {stem}")
            continue

        t_start = time.time()
        log(f"START {stem} ({path.stat().st_size / 1e6:.1f} MB)")
        try:
            gain_db, gain_reason = resolve_gain(path)
            log(f"  gain {gain_db:+.1f} dB ({gain_reason})")
            audio, duration = decode(path, gain_db)
            log(f"  audio {duration / 60:.1f} min decoded in {time.time() - t_start:.0f}s")

            if model is None:
                model = load_model(device, "float16")
                log("  model loaded on cuda/float16")

            t0 = time.time()
            try:
                segments, info = run_transcription(model, audio)
            except Exception as exc:
                log(f"  cuda failed ({type(exc).__name__}: {str(exc)[:160]}); retry cpu/int8")
                model = load_model("cpu", "int8")
                device = "cpu"
                segments, info = run_transcription(model, audio)

            if not segments and gain_db < MAX_AUTO_GAIN_DB:
                # Nothing detected: the level was still too low for the VAD.
                retry_gain = min(MAX_AUTO_GAIN_DB, gain_db + 30.0) or 30.0
                log(f"  empty result; retrying with gain {retry_gain:+.1f} dB")
                audio, _ = decode(path, retry_gain)
                segments, info = run_transcription(model, audio)
                if segments:
                    gain_db, gain_reason = retry_gain, "retry after empty result"

            elapsed = time.time() - t0
            if not segments:
                log(f"  EMPTY {stem}: no speech detected even at {gain_db:+.1f} dB; "
                    "output not written, rerun will retry")
                continue

            text = write_outputs(stem, path, duration, segments, info, gain_db, gain_reason)
            log(f"  DONE {stem}: {duration / 60:.1f} min audio in {elapsed / 60:.1f} min "
                f"({duration / max(elapsed, 1e-9):.1f}x), chars={len(text)}")
        except Exception as exc:
            log(f"  FAILED {stem}: {type(exc).__name__}: {str(exc)[:300]}")

    log("ALL FILES PROCESSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
