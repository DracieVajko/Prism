"""
Xiaomi-Nemotron-3.5-OCR-0 OCR engine for Prism PC.

Uses transformers to run the Xiaomi-OCR-0 model (Qwen3.5-0.8B based VLM)
for optical character recognition on images.

Model: https://huggingface.co/xiaomi-research/Xiaomi-OCR-0
Size: ~1.67 GB (bfloat16)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ocr_engine")

try:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class OCREngine:
    """Xiaomi Nemotron-3.5-OCR-0 OCR engine.

    Loads the Qwen3.5-0.8B based VLM model for optical character recognition.
    Supports multiple languages including Slovak, English, Czech, German, etc.
    """

    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or "pc/models/nemotron-ocr-0"
        self.processor = None
        self.model = None
        self._loaded = False

    def is_installed(self) -> bool:
        d = Path(self.model_dir)
        return (d / "model.safetensors").exists() and (d / "config.json").exists()

    def init(self) -> bool:
        """Load the OCR model. Returns True on success."""
        if self._loaded:
            return True

        if not _TORCH_AVAILABLE:
            logger.error("transformers/torch not installed - cannot load OCR model")
            return False

        if not self.is_installed():
            logger.warning(f"OCR model not installed at {self.model_dir}")
            return False

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq

            self.processor = AutoProcessor.from_pretrained(self.model_dir)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_dir,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            self._loaded = True
            logger.info(f"OCR model loaded from {self.model_dir}")
            return True
        except Exception as e:
            logger.error(f"OCR model init failed: {e}")
            self._loaded = False
            return False

    def release(self):
        """Release the model from memory."""
        self.processor = None
        self.model = None
        self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("OCR CUDA cache clear failed: %s", exc)

    def transcribe(self, image_path: str, language: str = "auto") -> str:
        """Perform OCR on an image file.

        Args:
            image_path: Path to the image file
            language: Language hint (auto, sk, en, cs, de, etc.) or "auto"

        Returns:
            Recognized text string
        """
        if self.model is None or self.processor is None:
            if not self.init():
                return ""

        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            # Build prompt based on language
            lang_map = {
                "sk": "Slovenčina",
                "cs": "Čeština",
                "en": "English",
                "de": "Deutsch",
                "fr": "Français",
                "es": "Español",
                "it": "Italiano",
                "pl": "Polski",
                "ru": "Русский",
                "uk": "Українська",
                "hu": "Magyar",
                "auto": None,
            }
            lang_name = lang_map.get(language, None)

            if lang_name:
                prompt = f"Transcribe the following image in {lang_name}. Return only the text content."
            else:
                prompt = "Transcribe the text in this image. Return only the text content."

            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            # Move to same device as model
            device = self.model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                    temperature=0.1,
                )

            # Decode only the generated tokens (skip input tokens)
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = output[0][input_len:]
            text = self.processor.decode(generated_tokens, skip_special_tokens=True)

            return text.strip()

        except Exception as e:
            logger.error(f"OCR transcription failed: {e}")
            return ""

    def transcribe_pil(self, image, language: str = "auto") -> str:
        """Perform OCR on a PIL Image object."""
        if self.model is None or self.processor is None:
            if not self.init():
                return ""

        try:
            import torch

            if image.mode != "RGB":
                image = image.convert("RGB")

            lang_map = {
                "sk": "Slovenčina",
                "cs": "Čeština",
                "en": "English",
                "de": "Deutsch",
                "fr": "Français",
                "es": "Español",
                "it": "Italiano",
                "pl": "Polski",
                "ru": "Русский",
                "uk": "Українська",
                "hu": "Magyar",
                "auto": None,
            }
            lang_name = lang_map.get(language, None)

            if lang_name:
                prompt = f"Transcribe the following image in {lang_name}. Return only the text content."
            else:
                prompt = "Transcribe the text in this image. Return only the text content."

            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            device = self.model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                    temperature=0.1,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_tokens = output[0][input_len:]
            text = self.processor.decode(generated_tokens, skip_special_tokens=True)

            return text.strip()

        except Exception as e:
            logger.error(f"OCR transcription failed: {e}")
            return ""


# Singleton instance
_ocr_engine = None


def get_ocr_engine() -> OCREngine:
    """Get the singleton OCR engine instance."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine()
    return _ocr_engine


def reset_ocr_engine():
    global _ocr_engine
    _ocr_engine = None
    _ocr_engine = OCREngine()
