"""OCR / document / image-understanding engines for Prism PC.

Each entry in OCR_SPECS is a model that can be downloaded from the Models tab
and selected on `POST /v1/images/ocr` (`model=`). Two families are supported:

  qwen-vl   Qwen2.5/Qwen3-VL style chat models. The Qwen chat template inserts
            the image tokens; without it the model sees zero image tokens.
  idefics3  Granite-Docling: takes a plain prompt and returns DocTags/Markdown.

Weights live in pc/models/<id>; download them with `download_models.py <id>`
or the WebUI catalog.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("ocr_engine")
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
    "paddleocr-vl-1.6": {
        "repo": "PaddlePaddle/PaddleOCR-VL-1.6",
        "family": "qwen-vl",
        "prompt": "OCR:",
        "label": "PaddleOCR-VL 1.6 (0.96B, layouts/tables/formulas)",
        "license": "Apache-2.0",
        "size_gb": 1.9,
    },
    "glm-ocr": {
        "repo": "zai-org/GLM-OCR",
        "family": "qwen-vl",
        "prompt": "Please extract the text from the image.",
        "label": "GLM-OCR 1.33B (MIT, tables/formulas, 8 languages)",
        "license": "MIT",
        "size_gb": 2.7,
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
        "family": "qwen-vl",
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


class OCREngine:
    def __init__(self, model_id: str = "nemotron-ocr-0", model_dir: str | None = None):
        self.model_id = model_id if model_id in OCR_SPECS else "nemotron-ocr-0"
        spec = OCR_SPECS[self.model_id]
        self.family = spec["family"]
        self.fixed_prompt = spec.get("prompt")
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
            self.processor = AutoProcessor.from_pretrained(
                self.model_dir, trust_remote_code=True)
            # auto = accelerate decides; cpu forces full CPU (works while the
            # GPU is busy with an LLM); cuda keeps everything on the GPU.
            device = (os.environ.get("PRISM_OCR_DEVICE") or "auto").strip().lower()
            device_map = device if device in ("cpu", "cuda") else "auto"
            self.model = _AutoVLM.from_pretrained(
                self.model_dir,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=True,
            )
            self._loaded = True
            logger.info("OCR model %s loaded from %s", self.model_id, self.model_dir)
            return True
        except Exception as e:
            logger.error("OCR model %s init failed: %s", self.model_id, e)
            self._loaded = False
            return False

    def release(self):
        self.processor = None
        self.model = None
        self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _prompt(self, task: str, language: str) -> str:
        task = (task or "ocr").strip().lower()
        if self.fixed_prompt:
            if task == "describe":
                raise ValueError(f"{self.model_id} only converts documents; "
                                 "use qwen3-vl-2b for description")
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
        # Both families need the chat template: it inserts the <image> token.
        # A bare prompt gives "0 <image> tokens and [0, 1] images".
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

    def transcribe_pil(self, image, language: str = "auto",
                       task: str = "ocr") -> str:
        if self.model is None or self.processor is None:
            if not self.init():
                return ""
        try:
            import torch

            if image.mode != "RGB":
                image = image.convert("RGB")
            prompt = self._prompt(task, language)
            inputs = self._inputs(image, prompt)
            device = self.model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = self.model.generate(
                    **inputs, max_new_tokens=4096, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            text = self.processor.decode(output[0][input_len:],
                                         skip_special_tokens=True)
            return text.strip()
        except Exception as e:
            logger.error("OCR transcription failed (%s): %s", self.model_id, e)
            return ""


_engines: dict[str, OCREngine] = {}


def get_ocr_engine(model_id: str = "nemotron-ocr-0") -> OCREngine:
    """Return a cached engine per model. Unknown ids fall back to nemotron."""
    mid = model_id if model_id in OCR_SPECS else "nemotron-ocr-0"
    if mid not in _engines:
        _engines[mid] = OCREngine(mid)
    return _engines[mid]


def reset_ocr_engine(model_id: str | None = None):
    if model_id is None:
        _engines.clear()
    else:
        _engines.pop(model_id, None)
