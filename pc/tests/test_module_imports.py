"""Import smoke tests for modules that currently have no direct test.

These are intentionally shallow: they fail if a module stops importing, which
is the minimum useful coverage for lazy/optional dependencies. Behavioural
tests for the engines live in their own test files.
"""
import importlib

MODULES = [
    "unified.action_extractor",
    "unified.action_handlers",
    "unified.ai_client",
    "unified.asr_common",
    "unified.document_extract",
    "unified.hf_asr",
    "unified.model_finder",
    "unified.restore_flux",
    "unified.silero_vad",
    "unified.tts_engine",
    "tools.download_gemma4",
    "tools.download_nemotron_asr",
    "legacy.run_diffusion_menu",
    "scripts.transcribe_income",
]


def test_modules_import():
    for name in MODULES:
        module = importlib.import_module(name)
        assert module.__name__ == name
