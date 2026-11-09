"""Kompletne porovnanie: whisper-turbo vs whisper-large vs nemotron-asr-0.6b vs gemma4-12B (safetensors + Q4_K_M GGUF)."""

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


def test_gemma4_cleanup(source_text: str, model_dir: str, label: str):
    """Gemma4 post-processing: cleanup + summarization."""
    print(f"\n=== Gemma4 {label} post-processing ===")
    try:
        gp.MODEL_DIR = Path(model_dir)
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
            save(f"gemma4-{label}-cleanup", result, "en", f"gemma4-{label}")
            gp.release()
    except Exception as e:
        print(f"CHYBA: {e}")
        import traceback
        traceback.print_exc()


def test_gemma4_gguf_cleanup(source_text: str):
    """Gemma4 12B Q4_K_M GGUF post-processing via llama-cpp-python."""
    print("\n=== Gemma4 12B Q4_K_M GGUF post-processing ===")
    try:
        from llama_cpp import Llama
        
        model_path = r"D:\AI\Prism\pc\models\gemma-4-12B-it-q4\gemma-4-12b-it-Q4_K_M.gguf"
        if not Path(model_path).exists():
            print("GGUF subor neexistuje, preskakujem")
            return
        
        print("Nacitavam Gemma4 12B Q4_K_M GGUF...")
        llm = Llama(
            model_path=model_path,
            n_ctx=8192,  # Dynamic context, not 256K
            n_threads=8,
            verbose=False,
        )
        print("Model nacitany uspesne")
        
        prompt = f"""<bos><start_of_turn>user
Clean up and summarize this meeting transcript. Fix grammar, add punctuation, structure with clear sections. Keep all important information. Reply with the cleaned text only.

{source_text}<end_of_turn>
<start_of_turn>model
"""
        
        result = llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            stop=["<end_of_turn>"],
            echo=False,
        )
        output = result["choices"][0]["text"].strip()
        print(f"Vysledok: {output[:300]}...")
        save("gemma4-12B-q4-gguf-cleanup", output, "en", "gemma4-12B-q4-gguf")
        
    except ImportError:
        print("llama-cpp-python nie je nainstalovany, preskakujem GGUF test")
        save("gemma4-12B-q4-gguf-ERROR", "llama-cpp-python not installed", "en", "error")
    except Exception as e:
        print(f"CHYBA: {e}")
        import traceback
        traceback.print_exc()
        save("gemma4-12B-q4-gguf-ERROR", str(e), "en", "error")


def main():
    print("=" * 60)
    print("POROVNANIE: turbo vs large vs nemotron vs gemma4-12B")
    print("=" * 60)

    # Test all three transcription engines
    test_whisper_turbo()
    test_whisper_large()
    test_nemotron()

    # Read whisper-turbo result as source for Gemma cleanup
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
        r"D:\AI\Prism\pc\models\gemma-4-12B-it",
        "12B-safetensors"
    )

    # Gemma4 12B Q4_K_M quantized GGUF
    test_gemma4_gguf_cleanup(source_text)

    print("\n" + "=" * 60)
    print("VYSLEDKY ULOZENE V:")
    print(OUTDIR)
    print("=" * 60)
    for name, path in results.items():
        print(f"  {name}: {path.name}")


if __name__ == "__main__":
    main()
