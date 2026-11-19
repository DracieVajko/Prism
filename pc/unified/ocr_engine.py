"""OCR / document / image-understanding engines for Prism PC.

Each entry in OCR_SPECS is a model that can be downloaded from the Models tab
and selected on `POST /v1/images/ocr` (`model=`). Supported families:

  qwen-vl     Qwen2.5/Qwen3-VL style chat models. The Qwen chat template
              inserts the image tokens; without it the model sees zero image
              tokens.
  native-vl   Models with a native transformers image-text class (GLM-OCR,
              LightOnOCR, PaddleOCR-VL, Qianfan-OCR). The processor embeds the
              image in the chat message and tokenises it in one call.
  idefics3    Granite-Docling: takes a plain prompt and returns DocTags/Markdown.

Weights live in pc/models/<id>; download them with `download_models.py <id>`
or the WebUI catalog.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
_HERE = Path(__file__).resolve().parent.parent  # pc/

try:
    import torch
    from transformers import AutoProcessor
    try:
        # transformers 5.x name; 4.x used AutoModelForVision2Seq.
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as _AutoVLM
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


OCR_SPECS = {
    "nemotron-ocr-0": {
        "repo": "xiaomi-research/Xiaomi-OCR-0",
        "family": "qwen-vl",
        "label": "Xiaomi Nemotron OCR 0.8B (fast, multilingual)",
        "license": "Apache-2.0",
        "size_gb": 1.7,
    },
    "granite-docling-258m": {
        "repo": "ibm-granite/granite-docling-258M",
        "family": "idefics3",
        "prompt": "Convert this page to docling.",
        "label": "Granite-Docling 258M (tiny, PDF -> DocTags/Markdown)",
        "license": "Apache-2.0",
        "size_gb": 0.5,
    },
    "qwen3-vl-2b": {
        "repo": "Qwen/Qwen3-VL-2B-Instruct",
        "family": "qwen-vl",
        "describe": True,
        "label": "Qwen3-VL 2B (OCR + objects/description, 32 languages)",
        "license": "Apache-2.0",
        "size_gb": 4.3,
    },
    "ovis-ocr2": {
        "repo": "ATH-MaaS/OvisOCR2",
        "family": "qwen-vl",
        "label": "OvisOCR2 853M (96.58 OmniDocBench v1.6)",
        "license": "Apache-2.0",
        "size_gb": 1.7,
    },
    "glm-ocr": {
        "repo": "zai-org/GLM-OCR",
        "family": "native-vl",
        "prompt": "Text Recognition:",
        "label": "GLM-OCR 0.9B (MIT, 94.62 OmniDocBench, tables/formulas)",
        "license": "MIT",
        "size_gb": 2.7,
    },
    "paddleocr-vl-1.6": {
        "repo": "PaddlePaddle/PaddleOCR-VL-1.6",
        "family": "native-vl",
        "prompt": "OCR:",
        "label": "PaddleOCR-VL 1.6 (0.96B, layouts/tables/formulas)",
        "license": "Apache-2.0",
        "size_gb": 1.9,
    },
    "lightonocr-2-1b": {
        "repo": "lightonai/LightOnOCR-2-1B",
        "family": "native-vl",
        "prompt": "",
        "document_only": True,
        "model_class": "LightOnOcrForConditionalGeneration",
        "processor_class": "LightOnOcrProcessor",
        "max_new_tokens": 1024,
        "label": "LightOnOCR 2 1B (fast end-to-end page OCR)",
        "license": "Apache-2.0",
        "size_gb": 2.0,
    },
    "firered-ocr": {
        "repo": "FireRedTeam/FireRed-OCR",
        "family": "qwen-vl",
        "label": "FireRed-OCR (Qwen3-VL-2B fine-tune)",
        "license": "Apache-2.0",
        "size_gb": 4.3,
    },
    "qianfan-ocr": {
        "repo": "baidu/Qianfan-OCR",
        "family": "native-vl",
        "prompt": "Parse this document to Markdown.",
        "label": "Qianfan-OCR 4.7B (192 languages, KIE, layout)",
        "license": "Apache-2.0",
        "size_gb": 9.5,
    },
}

LANG_NAMES = {
    "sk": "Slovenčina", "cs": "Čeština", "en": "English", "de": "Deutsch",
    "fr": "Français", "es": "Español", "it": "Italiano", "pl": "Polski",
    "ru": "Русский", "uk": "Українська", "hu": "Magyar",
}


def _has_weights(d: Path) -> bool:
    if not (d / "config.json").exists():
        return False
    if (d / "model.safetensors").exists():
        return True
    if (d / "model.safetensors.index.json").exists():
        return True
    return any(d.glob("*.safetensors"))


def resolve_ocr_model(model_id: str | None) -> str:
    """Pick the OCR model, with an installed-aware `auto` default.

    GLM-OCR is the UI-aware reader (reading order, sections, tables); Nemotron
    is the fast plain-text fallback. Explicit ids are returned unchanged so the
    endpoint can report an honest "not installed" for the requested model.
    """
    requested = (model_id or "auto").strip()
    if requested and requested.lower() != "auto":
        if requested in OCR_SPECS:
            return requested
        logger.warning("unknown OCR model %r; using auto", requested)
    for candidate in ("glm-ocr", "nemotron-ocr-0"):
        if _has_weights(_HERE / "models" / candidate):
            return candidate
    return "nemotron-ocr-0"


class OCREngine:
    def __init__(self, model_id: str = "auto", model_dir: str | None = None):
        self.model_id = resolve_ocr_model(model_id)
        spec = OCR_SPECS[self.model_id]
        self.family = spec["family"]
        self.fixed_prompt = spec.get("prompt")
        self.trust_remote_code = bool(spec.get("trust_remote_code", False))
        self.document_only = bool(spec.get("document_only", False)) or \
            self.fixed_prompt is not None
        self.supports_describe = bool(spec.get("describe", False))
        self.model_class = spec.get("model_class")
        self.processor_class = spec.get("processor_class")
        self.max_new_tokens = int(spec.get("max_new_tokens", 4096))
        self.model_dir = model_dir or str(_HERE / "models" / self.model_id)
        self.processor = None
        self.model = None
        self._loaded = False

    def is_installed(self) -> bool:
        return _has_weights(Path(self.model_dir))

    def init(self) -> bool:
        if self._loaded:
            return True
        if not _TORCH_AVAILABLE:
            logger.error("transformers/torch not installed - cannot load OCR model")
            return False
        if not self.is_installed():
            logger.warning("OCR model not installed at %s", self.model_dir)
            return False
        try:
            if self.processor_class:
                import transformers as _tf
                processor_cls = getattr(_tf, self.processor_class)
                self.processor = processor_cls.from_pretrained(self.model_dir)
            else:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_dir, trust_remote_code=self.trust_remote_code)
            # auto = accelerate decides; cpu forces full CPU (works while the
            # GPU is busy with an LLM); cuda keeps everything on the GPU.
            device = (os.environ.get("PRISM_OCR_DEVICE") or "auto").strip().lower()
            device_map = device if device in ("cpu", "cuda") else "auto"
            if self.model_class:
                # Some checkpoints declare a generic model_type (LightOnOCR 2
                # says "mistral3"), so the auto class builds the wrong module
                # and half the weights are dropped. Use the dedicated class.
                import transformers as _tf
                loader = getattr(_tf, self.model_class)
            else:
                loader = _AutoVLM
            self.model = loader.from_pretrained(
                self.model_dir,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=self.trust_remote_code,
            )
            self._loaded = True
            logger.info("OCR model %s loaded from %s", self.model_id, self.model_dir)
            return True
        except Exception as e:
            logger.error("OCR model %s init failed: %s", self.model_id, e)
            self._loaded = False
            return False

    def release(self) -> bool:
        had = self._loaded or self.model is not None or self.processor is not None
        self.processor = None
        self.model = None
        self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return had

    def _prompt(self, task: str, language: str) -> str:
        task = (task or "ocr").strip().lower()
        if task == "describe" and self.document_only:
            raise ValueError(f"{self.model_id} only converts documents; "
                             "use qwen3-vl-2b for description")
        if self.fixed_prompt is not None:
            return self.fixed_prompt
        if task == "describe":
            return ("Describe this image in detail. Cover the main objects, "
                    "people, layout, colours, and any visible text.")
        lang = LANG_NAMES.get(language)
        if lang:
            return (f"Transcribe the following image in {lang}. "
                    "Return only the text content.")
        return "Transcribe the text in this image. Return only the text content."

    def _inputs(self, image, prompt: str):
        if self.family == "native-vl":
            # Native image-text processors (GLM-OCR, LightOnOCR, PaddleOCR-VL,
            # Qianfan-OCR) want the PIL image inside the chat message and
            # tokenise it themselves. Passing images= separately loses the
            # image token for some of these processors.
            content = [{"type": "image", "image": image}]
            if prompt:
                content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        # Qwen-VL and Idefics3 need the chat template: it inserts the <image>
        # token. A bare prompt gives "0 <image> tokens and [0, 1] images".
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return self.processor(text=chat, images=[image], return_tensors="pt")

    def transcribe(self, image_path: str, language: str = "auto",
                   task: str = "ocr") -> str:
        """Open the file and delegate to transcribe_pil (one implementation)."""
        try:
            from PIL import Image

            return self.transcribe_pil(
                Image.open(image_path).convert("RGB"), language, task)
        except Exception as e:
            logger.error("OCR image open failed (%s): %s", self.model_id, e)
            return ""

    def generate_pil(self, image, prompt: str,
                     max_new_tokens: int | None = None) -> str:
        """Run one prompt against a PIL image and return the decoded text.

        Used by OCR (`transcribe_pil`) and by the VLM caption/chat path, so a
        single Qwen3-VL copy serves both endpoints.
        """
        if self.model is None or self.processor is None:
            if not self.init():
                raise RuntimeError(f"{self.model_id} failed to load")
        import torch

        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self._inputs(image, prompt)
        device = self.model.device
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        inputs = moved
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        text = self.processor.decode(output[0][input_len:],
                                     skip_special_tokens=True)
        return text.strip()

    def transcribe_pil(self, image, language: str = "auto",
                       task: str = "ocr") -> str:
        try:
            prompt = self._prompt(task, language)
            return self.generate_pil(image, prompt)
        except Exception as e:
            logger.error("OCR transcription failed (%s): %s", self.model_id, e)
            return ""


_engines: dict[str, OCREngine] = {}


def get_ocr_engine(model_id: str = "auto") -> OCREngine:
    """Return the engine for this model, keeping only one OCR model resident.

    Large OCR models plus a running LLM can exhaust 32 GB, so switching
    releases the previous engine instead of caching several.
    """
    mid = resolve_ocr_model(model_id)
    for other, eng in list(_engines.items()):
        if other != mid:
            eng.release()
            _engines.pop(other, None)
    if mid not in _engines:
        _engines[mid] = OCREngine(mid)
    return _engines[mid]


def reset_ocr_engine(model_id: str | None = None):
    if model_id is None:
        for eng in _engines.values():
            eng.release()
        _engines.clear()
    else:
        eng = _engines.pop(model_id, None)
        if eng is not None:
            eng.release()
