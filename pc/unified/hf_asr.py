"""Generic transformers ASR engines for Lenswright PC.

Transformers 5.x ships the classes for these models, so one pipeline covers
several families that would otherwise need NeMo/FunASR:

  qwen3-asr-0.6b / 1.7b   Qwen3ASRForConditionalGeneration (30 languages)
  qwencleo-asr-1.7b       Qwen3-ASR fine-tune (Egyptian Arabic + English)
  cohere-transcribe       CohereAsrForConditionalGeneration (14 languages)
  granite-speech-470m     GraniteSpeech5ForCTC (7 languages, fast)
  granite-speech-2b       GraniteSpeechForConditionalGeneration
  parakeet-v3 / v2        ParakeetForTDT (25 European / English)
  nb-whisper-medium       WhisperForConditionalGeneration (Norwegian)

Weights live in pc/models/<model_id>; sizes and licences are in ASR-MODELS.md.
The engine is explicit: `model=qwen3-asr-0.6b` etc. on /v1/audio/transcriptions.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_ID = "qwen3-asr-0.6b"
_HERE = Path(__file__).resolve().parent.parent  # pc/

SPECS = {
    "qwen3-asr-0.6b": {
        "label": "Qwen3-ASR 0.6B (30 languages)",
        "license": "Apache-2.0", "size_gb": 1.9,
    },
    "qwen3-asr-1.7b": {
        "label": "Qwen3-ASR 1.7B (30 languages)",
        "license": "Apache-2.0", "size_gb": 4.1,
    },
    "qwencleo-asr-1.7b": {
        "label": "QwenCleo ASR 1.7B (Egyptian Arabic + English)",
        "license": "Apache-2.0", "size_gb": 4.1,
    },
    "cohere-transcribe": {
        "label": "Cohere Transcribe (14 languages)",
        "license": "Apache-2.0", "size_gb": 4.1,
    },
    "granite-speech-470m": {
        "label": "Granite Speech 5.0 470M CTC (7 languages, fast)",
        "license": "Apache-2.0", "size_gb": 0.95,
    },
    "granite-speech-2b": {
        "label": "Granite Speech 5.0 2B (7 languages)",
        "license": "Apache-2.0", "size_gb": 5.0,
    },
    "parakeet-v3": {
        "label": "Parakeet TDT 0.6B v3 (25 European languages)",
        "license": "CC-BY-4.0", "size_gb": 5.7,
    },
    "parakeet-v2": {
        "label": "Parakeet TDT 0.6B v2 (English)",
        "license": "CC-BY-4.0", "size_gb": 2.5,
    },
    "parakeet-unified-en": {
        "label": "Parakeet Unified English (streaming)",
        "license": "see repo card", "size_gb": 2.5,
    },
    "nb-whisper-medium": {
        "label": "NB-Whisper Medium (Norwegian)",
        "license": "Apache-2.0", "size_gb": 3.1,
    },
}

_pipes: dict = {}
_models: dict = {}

LANG_NAMES = {
    "en": "English", "sk": "Slovak", "cs": "Czech", "de": "German",
    "fr": "French", "es": "Spanish", "it": "Italian", "pl": "Polish",
    "ru": "Russian", "uk": "Ukrainian", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "ar": "Arabic", "pt": "Portuguese", "nl": "Dutch",
    "tr": "Turkish", "hi": "Hindi", "th": "Thai", "vi": "Vietnamese",
}

QWEN_ASR_IDS = {"qwen3-asr-0.6b", "qwen3-asr-1.7b", "qwencleo-asr-1.7b"}


def _qwen_model(model_id: str):
    model = _models.get(model_id)
    if model is not None:
        return model
    import torch
    from qwen_asr import Qwen3ASRModel

    cuda = torch.cuda.is_available()
    model = Qwen3ASRModel.from_pretrained(
        str(_dir(model_id)),
        dtype=torch.bfloat16 if cuda else torch.float32,
        device_map="cuda:0" if cuda else "cpu",
        max_inference_batch_size=1,
        max_new_tokens=512,
    )
    _models[model_id] = model
    return model


def _dir(model_id: str) -> Path:
    return _HERE / "models" / model_id


def is_installed(model_id: str | None = None) -> bool:
    mid = model_id or MODEL_ID
    d = _dir(mid)
    if not (d / "config.json").exists():
        return False
    return (d / "model.safetensors").exists() or any(d.glob("*.safetensors"))


def enabled() -> bool:
    mode = (os.environ.get("LENSWRIGHT_ASR_ENGINE") or "").lower()
    return mode in ("hf", "transformers") or mode.startswith("qwen3-asr")


def release() -> bool:
    had = bool(_pipes or _models)
    _pipes.clear()
    _models.clear()
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return had


def _pipeline(model_id: str):
    pipe = _pipes.get(model_id)
    if pipe is not None:
        return pipe
    import torch
    from transformers import pipeline

    d = _dir(model_id)
    if not d.is_dir():
        raise FileNotFoundError(f"{model_id} not installed in {d}")
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if device == 0 else torch.float32
    logger.info("loading HF ASR %s on %s", model_id, "cuda" if device == 0 else "cpu")
    pipe = pipeline("automatic-speech-recognition", model=str(d),
                    device=device, torch_dtype=dtype)
    _pipes[model_id] = pipe
    return pipe


def transcribe(audio_path: str, language: str | None = None,
               model_id: str | None = None) -> tuple[str, str]:
    mid = model_id or MODEL_ID
    if mid in QWEN_ASR_IDS:
        # Qwen3-ASR ships its own transcribe() API; the generic ASR pipeline
        # passes the wrong tensor shapes to generate().
        model = _qwen_model(mid)
        lang_name = LANG_NAMES.get((language or "").lower()) if language else None
        results = model.transcribe(audio=str(audio_path), language=lang_name)
        if not results:
            return "", language or ""
        first = results[0]
        text = getattr(first, "text", "") or ""
        lang = getattr(first, "language", "") or (language or "")
        return text.strip(), str(lang)
    pipe = _pipeline(mid)
    kwargs = {"chunk_length_s": 30}
    if language and language not in ("auto", ""):
        kwargs["generate_kwargs"] = {"language": language}
    try:
        out = pipe(audio_path, **kwargs)
    except TypeError:
        out = pipe(audio_path, chunk_length_s=30)
    if isinstance(out, dict):
        text = (out.get("text") or "").strip()
    elif isinstance(out, list) and out and isinstance(out[0], dict):
        text = " ".join((x.get("text") or "").strip() for x in out).strip()
    else:
        text = str(out).strip()
    return text, language or ""
