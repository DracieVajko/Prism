"""External LLM providers for text post-processing.

The PC server does NOT download or manage any text-generation model — that is
the user's responsibility via their own LLM loader (Ollama, LM Studio, llama.cpp,
public API, etc.). This module only provides a thin client to call an external
endpoint that the user configures.

Configuration (environment variables):
  PRISM_LLM_PROVIDER   : ollama | lmstudio | openai | anthropic | custom | none
  PRISM_LLM_BASE_URL   : http://localhost:11434 (ollama), http://localhost:1234 (lmstudio), etc.
  PRISM_LLM_API_KEY    : API key for public providers (optional)
  PRISM_LLM_MODEL      : model name (e.g. "gemma4:12b", "qwen3.5-9b", "gpt-4o-mini")
  PRISM_LLM_ENDPOINT   : custom endpoint path (default: /api/generate or /v1/chat/completions)

Default prompts per operation (overridable via env vars PRISM_PROMPT_<OPERATION>):
  summary       : Summarize the text into structured notes with key points and action items.
  cleanup       : Fix grammar, spelling, and punctuation. Keep the meaning unchanged.
  grammar       : Fix grammar and spelling only. Do not rewrite or summarize.
  list          : Extract a bulleted list of key items from the text.
  translate     : Translate the text into the target language.
  rewrite       : Rewrite the text to be clearer and easier to read.
  meeting_notes : Summarize a meeting transcript with date, participants, topic,
                  decisions, action items, and open questions.
"""

import json
import logging
import os
from typing import Optional

import httpx

from unified.llm_monitor import acquire, loaded_models, release

logger = logging.getLogger("llm_providers")

# ---------------------------------------------------------------------------
# Default prompts per operation. Override with PRISM_PROMPT_<OPERATION> env var.
# ---------------------------------------------------------------------------
_DEFAULT_PROMPTS = {
    "summary": "Summarize the text into concise notes with key points.",
    "cleanup": "Fix grammar, spelling, and punctuation. Keep the meaning unchanged. Reply with the corrected text only.",
    "grammar": "Fix grammar and spelling only. Do not rewrite or summarize. Reply with the corrected text only.",
    "list": "Extract a bulleted list of the key items from the text. Reply with the list only.",
    "translate": "Translate the text into {target_lang}. Reply with the translation only.",
    "rewrite": "Rewrite the text to be clearer and easier to read. Keep the meaning and original language. Reply with the rewritten text only.",
    "meeting_notes": "Summarize this meeting transcript into structured notes with date, participants, topic, key decisions, summary points, and action items.",
}


def _get_prompt(operation: str) -> str:
    """Return the prompt for the given operation, honoring env overrides."""
    env_key = f"PRISM_PROMPT_{operation.upper()}"
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return env_val
    return _DEFAULT_PROMPTS.get(operation, _DEFAULT_PROMPTS["cleanup"])


# The UI says "summarize"; the prompt map key is "summary".
_OPERATION_ALIASES = {"summarize": "summary", "summarise": "summary"}


def _resolve_instruction(instruction: str, operation: Optional[str],
                         target_lang: str) -> str:
    """Pick the prompt to send: explicit instruction > named operation > cleanup.

    The UI posts the operation name in both ``post_process`` and
    ``post_process_operation``, so a known operation name must select the
    prompt rather than being sent to the model as the whole instruction.
    """
    instr = (instruction or "").strip()
    instr_key = _OPERATION_ALIASES.get(instr.lower(), instr.lower())
    op = _OPERATION_ALIASES.get((operation or "").strip().lower(),
                                (operation or "").strip().lower())

    if not instr:
        selected = op
    elif instr_key in _DEFAULT_PROMPTS:
        # A bare operation name ("translate") selects its prompt; it is not a
        # custom instruction to send to the model.
        selected = instr_key
    elif op:
        # A genuinely custom instruction overrides the operation default.
        return instr
    else:
        return instr

    resolved = _get_prompt(selected) if selected else _DEFAULT_PROMPTS["cleanup"]
    if target_lang and "{target_lang}" in resolved:
        resolved = resolved.format(target_lang=target_lang)
    return resolved


def _get_config() -> dict:
    return {
        "provider": os.environ.get("PRISM_LLM_PROVIDER", "none").strip().lower(),
        "base_url": os.environ.get("PRISM_LLM_BASE_URL", "").strip().rstrip("/"),
        "api_key": os.environ.get("PRISM_LLM_API_KEY", "").strip(),
        "model": os.environ.get("PRISM_LLM_MODEL", "gemma4:12b").strip(),
        "endpoint": os.environ.get("PRISM_LLM_ENDPOINT", "").strip(),
    }


def _build_payload(provider: str, model: str, instruction: str, text: str,
                   target_lang: str, endpoint: str) -> tuple[dict, str]:
    """Return (payload, url) for the given provider."""
    full_prompt = (
        f"{instruction}\nTarget language: {target_lang}\n\nText:\n{text}\n\n"
        f"Reply with the result only."
    )

    if provider == "ollama":
        url = f"{endpoint or '/api/generate'}"
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
        }
        return payload, url

    if provider == "lmstudio":
        url = f"{endpoint or '/v1/chat/completions'}"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": full_prompt},
            ],
            "temperature": 0.7,
            "stream": False,
        }
        return payload, url

    if provider in ("openai", "anthropic", "custom"):
        url = f"{endpoint or '/v1/chat/completions'}"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": full_prompt},
            ],
            "temperature": 0.7,
            "stream": False,
        }
        return payload, url

    # Default: OpenAI-compatible
    url = f"{endpoint or '/v1/chat/completions'}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": full_prompt},
        ],
        "temperature": 0.7,
        "stream": False,
    }
    return payload, url


def process_with_external_llm(
    text: str,
    instruction: str = "cleanup",
    target_lang: str = "en",
    operation: Optional[str] = None,
) -> str:
    """Send text to an external LLM provider and return the processed result.

    Args:
        text: Source text to process.
        instruction: Custom instruction (overrides operation default if set).
        target_lang: Target language for translation/summary.
        operation: Named operation (summary, cleanup, grammar, list, translate,
                   rewrite, meeting_notes). Uses default prompt if instruction
                   is not explicitly customized.

    Returns the original text unchanged if no provider is configured or the
    call fails — the caller should treat this as a best-effort enhancement.
    """
    cfg = _get_config()
    provider = cfg["provider"]

    if provider in ("none", "", "disabled"):
        logger.info("No external LLM provider configured; skipping post-processing")
        return text

    if not cfg["base_url"]:
        logger.warning("LLM provider %s configured but no base URL set", provider)
        return text

    # Local AI resource gate: respect the configured concurrent-model limit.
    # Wait for a slot instead of failing when another model is busy; only
    # fall back (return original text) after the configured timeout.
    model = cfg["model"]
    # A model that was already resident belongs to the user, not to this call.
    before_loaded = model in loaded_models()
    acquired = acquire(model)
    if not acquired:
        logger.warning("Could not acquire an LLM slot in time; skipping post-processing")
        return text

    resolved_instruction = _resolve_instruction(instruction, operation, target_lang)

    try:
        payload, path = _build_payload(
            provider, cfg["model"], resolved_instruction, text, target_lang, cfg["endpoint"]
        )
        url = cfg["base_url"] + path

        headers = {"Content-Type": "application/json"}
        if cfg["api_key"]:
            headers["Authorization"] = f"Bearer {cfg['api_key']}"

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # Extract text from various provider response shapes
        result = _extract_response_text(provider, data)
        if result:
            return result.strip()
        logger.warning("External LLM returned empty response")
        return text

    except Exception as e:
        logger.warning("External LLM call failed: %s", str(e)[:200])
        return text
    finally:
        if acquired and not before_loaded:
            try:
                release(model)
            except Exception:
                pass


def _extract_response_text(provider: str, data: dict) -> str:
    """Extract generated text from provider-specific response shapes."""
    if provider == "ollama":
        return data.get("response", "")

    # OpenAI-compatible (LM Studio, OpenAI, Anthropic-compatible, custom)
    choices = data.get("choices")
    if isinstance(choices, list) and len(choices) > 0:
        msg = choices[0].get("message", {})
        if isinstance(msg, dict):
            return msg.get("content", "")
        text_val = choices[0].get("text", "")
        if text_val:
            return text_val

    # Fallback: look for common fields
    for key in ("content", "text", "output", "result", "completion"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val

    return ""
