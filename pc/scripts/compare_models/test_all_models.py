"""Complete transcription test: ALL models — Whisper small/turbo/large, Nemotron 0.6B, Gemma4 post-processing."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unified.audio_splitter import get_audio_duration
from unified.ffmpeg_tools import extract_audio
from unified import nemotron_asr
from unified.transcription_utils import (
    format_transcription, split_text_into_chunks, validate_video_file,
)
from unified_server import _transcribe_with_whisper

VIDEO = Path(r"D:\AI\Prism\testdata\video\test_1min.mp4")
AUDIO_OUT = Path(r"D:\AI\Prism\testdata\video\test_1min_audio.wav")

results = {}


def extract_audio_from_video():
    """Extract audio from video to 16kHz mono WAV."""
    print("=" * 70)
    print("KROK 1: Extrakcia audio z videa")
    print("=" * 70)
    result_path, info = extract_audio(VIDEO, output_path=AUDIO_OUT, audio_format="wav")
    dur = get_audio_duration(result_path)
    print(f"Video: {VIDEO.name}")
    print(f"Audio: {result_path.name}")
    print(f"Dlzka: {dur:.1f}s")
    print(f"Info: {info.get('format', {})}")
    return result_path


def test_whisper(model_variant: str):
    """Test faster-whisper model."""
    print("\n" + "=" * 70)
    print(f"KROK 2: Whisper {model_variant}")
    print("=" * 70)
    try:
        text, lang, segments = _transcribe_with_whisper(
            str(AUDIO_OUT), model_variant, None
        )
        print(f"Model: whisper-{model_variant}")
        print(f"Jazyk: {lang}")
        print(f"Segmentov: {len(segments)}")
        print(f"Text (prvych 500 znakov): {text[:500]}")
        results[f"whisper-{model_variant}"] = (text, lang, segments)
        return text, lang, segments
    except Exception as e:
        print(f"CHYBA whisper-{model_variant}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_nemotron():
    """Test Nemotron ASR 0.6B."""
    print("\n" + "=" * 70)
    print("KROK 3: Nemotron ASR 0.6B")
    print("=" * 70)
    try:
        text, lang = nemotron_asr.transcribe(
            str(AUDIO_OUT), None, "nemotron-asr-0.6b"
        )
        segments = [{"text": text, "start": 0.0, "end": 60.0}]
        print(f"Model: nemotron-asr-0.6b")
        print(f"Jazyk: {lang}")
        print(f"Segmentov: {len(segments)}")
        print(f"Text (prvych 500 znakov): {text[:500]}")
        results["nemotron-asr-0.6b"] = (text, lang, segments)
        return text, lang, segments
    except Exception as e:
        print(f"CHYBA nemotron-asr-0.6b: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_output_formats(text: str, lang: str, engine: str):
    """Test all output formats: txt, md, srt, json."""
    print("\n" + "=" * 70)
    print(f"KROK 4: Vystupne formaty ({engine})")
    print("=" * 70)
    segments = [{"text": text, "start": 0.0, "end": 60.0}]
    for fmt in ("txt", "md", "srt", "json"):
        formatted = format_transcription(segments, fmt, lang)
        print(f"\n--- {fmt.upper()} ---")
        print(formatted[:600] + ("..." if len(formatted) > 600 else ""))


def test_text_chunking(text: str, max_chars: int = 2000):
    """Test text chunking by char limit."""
    print("\n" + "=" * 70)
    print(f"KROK 5: Chunking textu (max_chars={max_chars})")
    print("=" * 70)
    chunks = split_text_into_chunks(text, max_chars=max_chars)
    print(f"Pvodna dlzka: {len(text)} znakov")
    print(f"Pocet chunkov: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {len(chunk)} znakov - {chunk[:80]}...")


def test_external_llm_postprocess(text: str):
    """Test external LLM post-processing for meeting summarization."""
    print("\n" + "=" * 70)
    print("KROK 6: External LLM post-processing (sumarizacia stretnutia)")
    print("=" * 70)
    try:
        from unified.llm_providers import process_with_external_llm
        result = process_with_external_llm(text, operation="meeting_notes")
        print(f"Vysledok (prvych 1000 znakov): {result[:1000]}")
        results["external-llm-summary"] = result
        return result
    except Exception as e:
        print(f"CHYBA external LLM: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 70)
    print("PRISM — Kompletny test vsetkych modelov")
    print("=" * 70)

    # Validate video
    ok, err = validate_video_file(str(VIDEO))
    if not ok:
        print(f"VIDEO VALIDACIA ZLYHALA: {err}")
        return
    print(f"Video validovane: {VIDEO.name}")

    # Step 1: Extract audio
    audio_path = extract_audio_from_video()

    # Step 2: Test all whisper variants
    for variant in ("small", "turbo", "large"):
        test_whisper(variant)

    # Step 3: Test Nemotron
    test_nemotron()

    # Step 4-5: Test output formats + chunking with first successful result
    for engine, (text, lang, segments) in results.items():
        if engine.startswith("whisper") or engine.startswith("nemotron"):
            test_output_formats(text, lang, engine)
            test_text_chunking(text, max_chars=2000)
            break

    # Step 6: External LLM post-processing
    if results:
        first_text = list(results.values())[0][0]
        test_external_llm_postprocess(first_text)

    # Cleanup
    try:
        AUDIO_OUT.unlink(missing_ok=True)
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("VSETKY TESTY DOKONCENE!")
    print("=" * 70)
    print(f"Otestovanych enginov: {len(results)}")
    for name in results:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
