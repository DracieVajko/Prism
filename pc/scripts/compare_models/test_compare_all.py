"""Run ALL transcription models and save results to files for comparison."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unified_server import _transcribe_with_whisper
from unified import nemotron_asr
import unified.gemma_postprocess as gp

AUDIO = Path(r"D:\AI\Prism\testdata\video\test_1min_audio.wav")
OUTDIR = Path(r"D:\AI\Prism\testdata\transcription_comparison")
OUTDIR.mkdir(parents=True, exist_ok=True)

results = {}


def save(name: str, text: str, lang: str = "", segments=None):
    """Save transcription result to file."""
    p = OUTDIR / f"{name}.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"=== {name} ===\n")
        f.write(f"Jazyk: {lang}\n")
        f.write(f"Segmenty: {len(segments) if segments else 'n/a'}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)
        f.write("\n")
    results[name] = p
    print(f"Ulozeno: {p}")


def test_whisper(variant: str):
    """Test one Whisper variant."""
    print(f"\n=== Whisper {variant} ===")
    try:
        text, lang, segments = _transcribe_with_whisper(str(AUDIO), variant, None)
        print(f"Jazyk: {lang}, Segmenty: {len(segments)}")
        print(f"Text: {text[:200]}...")
        save(f"whisper-{variant}", text, lang, segments)
    except Exception as e:
        print(f"CHYBA: {e}")
        save(f"whisper-{variant}-ERROR", str(e))


def test_nemotron():
    """Test Nemotron ASR 0.6B."""
    print(f"\n=== Nemotron ASR 0.6B ===")
    try:
        text, lang = nemotron_asr.transcribe(str(AUDIO), None, "nemotron-asr-0.6b")
        segments = [{"text": text, "start": 0.0, "end": 60.0}]
        print(f"Jazyk: {lang}, Segmenty: {len(segments)}")
        print(f"Text: {text[:200]}...")
        save("nemotron-asr-0.6b", text, lang, segments)
    except Exception as e:
        print(f"CHYBA: {e}")
        save("nemotron-asr-0.6b-ERROR", str(e))


def test_gemma4(variant_dir: str, name: str):
    """Test Gemma4 post-processing with a specific model variant."""
    print(f"\n=== Gemma4 {name} post-processing ===")
    try:
        # Redirect model dir
        gp.MODEL_DIR = Path(variant_dir)
        gp._processor = None
        gp._model = None
        gp._device = None

        import os
        os.environ["PRISM_GEMMA_DTYPE"] = "bfloat16"

        ok = gp.init(device="cpu")
        print(f"Inicializacia: {'uspesna' if ok else 'ZLYHALA'}")

        if ok:
            # Read whisper-turbo transcription as source text
            src = OUTDIR / "whisper-turbo.txt"
            if src.exists():
                with open(src, "r", encoding="utf-8") as f:
                    content = f.read()
                # Extract just the text part (after the header)
                text_start = content.find("=" * 60) + len("=" * 60) + 2
                source_text = content[text_start:].strip()
            else:
                source_text = "all of that and I actually paid someone else so I needed a bit of money and we got our first internet page up or website shall I say and after a few days it was doing quite well it wasnt selling there was no e-commerce on there it was purely to get people to the shop to see what we sold so they could phone us email us and all that sort of thing but within about two years that was 50% of my business was online now all my other friends were opening up multiple shops going to different locations"

            instruction = (
                "Summarize this meeting transcript into structured notes with date, "
                "participants, topic, key decisions, summary points, and action items."
            )
            result = gp.process(source_text, instruction=instruction, target_lang="en")
            print(f"Vysledok: {result[:300]}...")
            save(f"gemma4-{name}-summary", result, "en", None)
            gp.release()
    except Exception as e:
        print(f"CHYBA: {e}")
        import traceback
        traceback.print_exc()
        save(f"gemma4-{name}-ERROR", str(e))


def main():
    print("=" * 60)
    print("KOMPLETNY POROVNANIE VSECH MODELOV")
    print("=" * 60)

    # Test all whisper variants
    for variant in ("tiny", "small", "turbo", "large"):
        test_whisper(variant)

    # Test Nemotron
    test_nemotron()

    # Test Gemma4 post-processing with both variants
    test_gemma4(
        r"D:\AI\Prism\pc\models\gemma-4-E2B-it", "E2B-2B"
    )
    test_gemma4(
        r"D:\AI\Prism\pc\models\gemma-4-12B-it", "12B"
    )

    # Summary
    print("\n" + "=" * 60)
    print("VYSLEDKY ULOZENE V:")
    print(OUTDIR)
    print("=" * 60)
    for name, path in results.items():
        print(f"  {name}: {path.name}")


if __name__ == "__main__":
    main()
