"""Post-processing prompt selection — operations must not be sent literally."""
from unified import llm_providers


def test_operation_selects_its_prompt():
    prompt = llm_providers._resolve_instruction("", "cleanup", "en")
    assert "Fix grammar, spelling, and punctuation" in prompt


def test_bare_operation_name_is_a_selector_not_an_instruction(monkeypatch):
    # The WebUI sends "translate" in both fields; the model must get the
    # translate prompt with the target language filled in.
    monkeypatch.delenv("PRISM_PROMPT_TRANSLATE", raising=False)
    prompt = llm_providers._resolve_instruction("translate", "translate", "sk")
    assert "Translate the text into sk" in prompt


def test_ui_summarize_alias_maps_to_summary(monkeypatch):
    monkeypatch.delenv("PRISM_PROMPT_SUMMARY", raising=False)
    prompt = llm_providers._resolve_instruction("summarize", "summarize", "en")
    assert "Summarize the text" in prompt


def test_custom_instruction_wins_over_operation():
    prompt = llm_providers._resolve_instruction("Reply in one sentence", "summary", "en")
    assert prompt == "Reply in one sentence"


def test_empty_falls_back_to_cleanup_prompt():
    assert llm_providers._resolve_instruction("", None, "en") == \
        llm_providers._DEFAULT_PROMPTS["cleanup"]


def test_env_override_is_used(monkeypatch):
    monkeypatch.setenv("PRISM_PROMPT_CLEANUP", "Only fix commas.")
    assert llm_providers._resolve_instruction("", "cleanup", "en") == "Only fix commas."
