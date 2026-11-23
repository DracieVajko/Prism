"""Transcribe phone recordings synced into the local Recordings\\Record folder.

Reuses the pipeline in transcribe_income.py (faster-whisper large, auto gain,
CUDA float16 with CPU int8 fallback) and writes, next to each recording:

    <name>.txt         plain text; a non-empty one makes a re-run skip the file
    <name>.srt         subtitles with timestamps
    <name>.json        metadata plus the segments
    <name>.digest.md   speech statistics, long pauses and the transcript split
                       into timestamped chunks small enough for one AI context

A transcript that fails the deterministic quality gate in
unified.transcription_utils is transcribed again with stronger decoding
settings, cheapest first. Every rejected attempt is kept as <name>.roughN.* so
nothing is lost, and the best attempt becomes the real transcript.

Usage:
    python pc/scripts/transcribe_recordings.py               # all new recordings
    python pc/scripts/transcribe_recordings.py 20261121      # only matching names
"""
import json
import sys
import time
from pathlib import Path

import transcribe_income as ti

ROOT = Path(r"C:\Users\DracieVajko\Music\Recordings\Record")
OUT_DIR = ROOT
ti.OUT_DIR = OUT_DIR

# `unified` lives in pc/, one level above this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified.meeting_digest import render_digest  # noqa: E402
from unified.transcription_utils import assess_transcript_quality  # noqa: E402

# Decoding attempts, cheapest first: (label, beam, vad, extra gain in dB).
# The base attempt uses whatever PRISM_ASR_BEAM / PRISM_ASR_VAD say.
ATTEMPTS = (
    ("base", ti.BEAM_SIZE, ti.VAD_FILTER, 0.0),
    ("beam5", 5, ti.VAD_FILTER, 0.0),
    ("beam5-novad", 5, False, 0.0),
    ("beam5-novad-loud", 5, False, 30.0),
)


def transcribe(model_box, audio, beam, vad):
    """Run faster-whisper, falling back to CPU int8 when CUDA fails."""
    previous = (ti.BEAM_SIZE, ti.VAD_FILTER)
    ti.BEAM_SIZE, ti.VAD_FILTER = beam, vad
    try:
        try:
            return ti.run_transcription(model_box["model"], audio)
        except Exception as exc:
            ti.log(f"  cuda failed ({type(exc).__name__}: {str(exc)[:160]}); "
                   "retry cpu/int8")
            model_box["model"] = ti.load_model("cpu", "int8")
            model_box["device"] = "cpu"
            return ti.run_transcription(model_box["model"], audio)
    finally:
        ti.BEAM_SIZE, ti.VAD_FILTER = previous


def rank(attempt):
    """Order attempts worst to best: usable first, then variety, then volume."""
    quality = attempt["quality"]
    return (quality["ok"], quality["unique_ratio"], quality["words"])


def write_recording(model_box, path, base_gain):
    """Transcribe one recording and write every artifact. Returns a result row."""
    stem = path.stem
    decoded_gain = base_gain
    audio, duration = ti.decode(path, decoded_gain)
    ti.log(f"  audio {duration / 60:.1f} min decoded")

    attempts = []
    for label, beam, vad, extra_gain in ATTEMPTS:
        if extra_gain:
            # Stronger gain than the level-based one, for speech the VAD missed.
            wanted_gain = min(ti.MAX_AUTO_GAIN_DB, decoded_gain + extra_gain) or extra_gain
        else:
            wanted_gain = decoded_gain
        if wanted_gain != decoded_gain:
            audio, duration = ti.decode(path, wanted_gain)
            decoded_gain = wanted_gain
            ti.log(f"  re-decoded at {decoded_gain:+.1f} dB for {label}")

        started = time.time()
        segments, info = transcribe(model_box, audio, beam, vad)
        quality = assess_transcript_quality(segments, duration)
        ti.log(f"  {label}: {len(segments)} segments, {quality['words']} words, "
               f"{quality['wpm']:.0f} wpm, unique {quality['unique_ratio']:.2f}, "
               f"{'ok' if quality['ok'] else 'rejected ' + '/'.join(quality['reasons'])} "
               f"({time.time() - started:.0f}s)")
        attempts.append({
            "label": label,
            "segments": segments,
            "info": info,
            "gain_db": decoded_gain,
            "quality": quality,
        })
        if quality["ok"]:
            break

    usable = [a for a in attempts if a["segments"]]
    if not usable:
        ti.log(f"  EMPTY {path.name}: no speech detected in any attempt; "
               "nothing written, a rerun will try again")
        return {"file": str(path), "txt": None, "digest": None, "status": "empty",
                "attempts": [{"label": a["label"], **a["quality"]} for a in attempts]}

    winner = max(usable, key=rank)
    rows = []
    for index, attempt in enumerate(attempts, 1):
        if not attempt["segments"]:
            rows.append({"label": attempt["label"], **attempt["quality"],
                         "written": None})
            continue
        target = stem if attempt is winner else f"{stem}.rough{index}"
        text = ti.write_outputs(target, path, duration, attempt["segments"],
                                attempt["info"], attempt["gain_db"],
                                f"attempt {attempt['label']}")
        ti.log(f"  wrote {target}.txt ({len(text)} chars)")
        rows.append({"label": attempt["label"], **attempt["quality"],
                     "written": f"{target}.txt"})

    digest_path = OUT_DIR / f"{stem}.digest.md"
    digest_path.write_text(
        render_digest(
            name=stem,
            segments=winner["segments"],
            duration_s=duration,
            language=getattr(winner["info"], "language", None),
        ),
        encoding="utf-8",
    )

    return {
        "file": str(path),
        "txt": str(OUT_DIR / f"{stem}.txt"),
        "srt": str(OUT_DIR / f"{stem}.srt"),
        "json": str(OUT_DIR / f"{stem}.json"),
        "digest": str(digest_path),
        "status": "done" if winner["quality"]["ok"] else "done-needs-review",
        "chosen": winner["label"],
        "attempts": rows,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ti.TMP_DIR.mkdir(parents=True, exist_ok=True)
    ti.log(f"cuda dll dirs: {ti.add_cuda_dlls()}")

    files = sorted(
        p for p in ROOT.glob("*") if p.is_file() and p.suffix.lower() in ti.EXTS
    )
    if len(sys.argv) > 1:
        needle = sys.argv[1].lower()
        files = [p for p in files if needle in p.name.lower()]
    ti.log(f"{len(files)} recordings, model=large-v3")

    model_box = {"model": None, "device": "cuda"}
    results = []
    for path in files:
        out_txt = OUT_DIR / f"{path.stem}.txt"
        if out_txt.exists() and out_txt.stat().st_size > 0:
            ti.log(f"skip (done): {path.name}")
            results.append({"file": str(path), "txt": str(out_txt),
                            "status": "skipped"})
            continue

        ti.log(f"START {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        try:
            gain_db, gain_reason = ti.resolve_gain(path)
            ti.log(f"  gain {gain_db:+.1f} dB ({gain_reason})")
            if model_box["model"] is None:
                model_box["model"] = ti.load_model(model_box["device"], "float16")
                ti.log("  model loaded on cuda/float16")
            results.append(write_recording(model_box, path, gain_db))
        except Exception as exc:
            ti.log(f"  FAILED {path.name}: {type(exc).__name__}: {str(exc)[:300]}")
            results.append({"file": str(path), "txt": None,
                            "status": f"failed: {type(exc).__name__}"})

    ti.log("ALL RECORDINGS PROCESSED")
    print(json.dumps(results, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
