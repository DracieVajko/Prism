"""Porovnanie: whisper-turbo vs whisper-large vs nemotron-asr-0.6b vs gemma4-12B (post-processing/cleanup)."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unified_server import _transcribe_with_whisper
from unified import nemotron_asr
import unified.gemma_postprocess as gp

AUDIO = Path(r"D:\AI\Prism\testdata\video\test_1min_audio.wav")
OUTDIR = Path(r"D:\AI\Prism\testdata\transcription_comparison_v2")
OUTDIR.mkdir(parents=True, exist_ok=True)

results = {}


def save(name: str, text: str, lang: str = "", engine: str = ""):
    p = OUTDIR / f"{name}.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"=== {name} ===\n")
        f.write(f"Engine: {engine}\n")
        f.write(f"Jazyk: {lang}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)
        f.write("\n")
    results[name] = p
    print(f"Ulozeno: {p.name}")


def test_whisper_turbo():
    print("\n=== Whisper TURBO ===")
    text, lang, segments = _transcribe_with_whisper(str(AUDIO), "turbo", None)
    print(f"Jazyk: {lang}, Segmenty: {len(segments)}")
    print(f"Text: {text[:200]}...")
    save("whisper-turbo", text, lang, "faster-whisper-turbo")


def test_whisper_large():
    print("\n=== Whisper LARGE ===")
    text, lang, segments = _transcribe_with_whisper(str(AUDIO), "large", None)
    print(f"Jazyk: {lang}, Segmenty: {len(segments)}")
    print(f"Text: {text[:200]}...")
    save("whisper-large", text, lang, "faster-whisper-large")


def test_nemotron():
    print("\n=== Nemotron ASR 0.6B ===")
    text, lang = nemotron_asr.transcribe(str(AUDIO), None, "nemotron-asr-0.6b")
    segments = [{"text": text, "start": 0.0, "end": 60.0}]
    print(f"Jazyk: {lang}, Segmenty: {len(segments)}")
    print(f"Text: {text[:200]}...")
    save("nemotron-asr-0.6b", text, lang, "nemotron-asr-0.6b")


def test_gemma4_cleanup(source_text: str, model_variant: str):
    """Gemma4 post-processing: cleanup + summarization."""
    print(f"\n=== Gemma4 {model_variant} post-processing ===")
    try:
        gp.MODEL_DIR = Path(model_variant)
        gp._processor = None
        gp._model = None
        gp._device = None

        os.environ["PRISM_GEMMA_DTYPE"] = "bfloat16"
        ok = gp.init(device="cpu")
        print(f"Inicializacia: {'uspesna' if ok else 'ZLYHALA'}")

        if ok:
            instruction = (
                "Clean up and summarize this meeting transcript. "
                "Fix grammar, add punctuation, structure with clear sections. "
                "Keep all important information. Reply with the cleaned text only."
            )
            result = gp.process(source_text, instruction=instruction, target_lang="en")
            print(f"Vysledok: {result[:300]}...")
            save(f"gemma4-{model_variant.name}-cleanup", result, "en", f"gemma4-{model_variant.name}")
            gp.release()
    except Exception as e:
        print(f"CHYBA: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 60)
    print("POROVNANIE: turbo vs large vs nemotron vs gemma4-12B")
    print("=" * 60)

    # Test all three transcription engines
    test_whisper_turbo()
    test_whisper_large()
    test_nemotron()

    # Gemma4 cleanup using whisper-turbo output as source
    # Read whisper-turbo result
    turbo_file = OUTDIR / "whisper-turbo.txt"
    if turbo_file.exists():
        with open(turbo_file, "r", encoding="utf-8") as f:
            content = f.read()
        text_start = content.find("=" * 60) + len("=" * 60) + 2
        source_text = content[text_start:].strip()
    else:
        source_text = ""

    # Gemma4 12B safetensors (already downloaded)
    test_gemma4_cleanup(
        source_text,
        r"D:\AI\Prism\pc\models\gemma-4-12B-it"
    )

    # Gemma4 12B Q4_K_M quantized (being downloaded)
    q4_dir = r"D:\AI\Prism\pc\models\gemma-4-12B-it-q4"
    if Path(q4_dir).exists():
        test_gemma4_cleanup(
            source_text,
            q4_dir
        )

    print("\n" + "=" * 60)
    print("VYSLEDKY ULOZENE V:")
    print(OUTDIR)
    print("=" * 60)
    for name, path in results.items():
        print(f"  {name}: {path.name}")


if __name__ == "__main__":
    main()
