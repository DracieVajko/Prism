"""Registry checks for the PC OCR engines.

These tests do not load weights; they pin the model ids, families and prompt
rules that the HTTP endpoint relies on.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unified import ocr_engine  # noqa: E402
import download_models  # noqa: E402


def test_catalog_and_engine_agree():
    engine_ids = set(ocr_engine.OCR_SPECS)
    catalog_ids = {t["id"] for t in download_models.OCR_TASKS}
    # Every offered engine must be downloadable, and every catalogued OCR
    # model must route to an engine (no dead catalog rows).
    assert engine_ids == catalog_ids


def test_families_are_known():
    known = {"qwen-vl", "native-vl", "idefics3"}
    for model_id, spec in ocr_engine.OCR_SPECS.items():
        assert spec["family"] in known, model_id


@pytest.mark.parametrize("model_id", [
    "glm-ocr", "paddleocr-vl-1.6", "lightonocr-2-1b", "qianfan-ocr",
])
def test_native_models_use_native_family(model_id):
    spec = ocr_engine.OCR_SPECS[model_id]
    assert spec["family"] == "native-vl"
    assert spec["license"]


def test_lighton_uses_dedicated_class_and_empty_prompt():
    spec = ocr_engine.OCR_SPECS["lightonocr-2-1b"]
    assert spec["model_class"] == "LightOnOcrForConditionalGeneration"
    assert spec["processor_class"] == "LightOnOcrProcessor"
    assert spec["prompt"] == ""
    assert spec["document_only"] is True


def test_document_models_reject_description():
    for model_id in ("glm-ocr", "paddleocr-vl-1.6", "lightonocr-2-1b",
                     "qianfan-ocr", "granite-docling-258m"):
        engine = ocr_engine.OCREngine(model_id)
        with pytest.raises(ValueError):
            engine._prompt("describe", "auto")


def test_qwen3_vl_can_describe():
    engine = ocr_engine.OCREngine("qwen3-vl-2b")
    assert engine.supports_describe is True
    prompt = engine._prompt("describe", "auto")
    assert "Describe this image" in prompt


def test_only_qwen3_vl_offers_description():
    for model_id, spec in ocr_engine.OCR_SPECS.items():
        assert bool(spec.get("describe", False)) == (model_id == "qwen3-vl-2b")


def test_unknown_model_falls_back_to_nemotron():
    engine = ocr_engine.OCREngine("does-not-exist")
    assert engine.model_id == "nemotron-ocr-0"
