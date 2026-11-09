"""External LLM providers for text post-processing.

The PC server does NOT download or manage any text-generation model — that is
the user's responsibility via their own LLM loader (Ollama, LM Studio, llama.cpp,
public API, etc.). This module only provides a thin client to call an external
endpoint that the user configures.

Configuration (environment variables or AppConfig):
  PRISM_LLM_PROVIDER   : ollama | lmstudio | openai | anthropic | custom | none
  PRISM_LLM_BASE_URL   : http://localhost:11434 (ollama), http://localhost:1234 (lmstudio), etc.
  PRISM_LLM_API_KEY    : API key for public providers (optional)
  PRISM_LLM_MODEL      : model name (e.g. "gemma4:12b", "qwen3.5-9b", "gpt-4o-mini")
  PRISM_LLM_ENDPOINT   : custom endpoint path (default: /api/generate or /v1/chat/completions)
"""

import json
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("llm_providers")


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
) -> str:
    """Send text to an external LLM provider and return the processed result.

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

    try:
        payload, path = _build_payload(
            provider, cfg["model"], instruction, text, target_lang, cfg["endpoint"]
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
        text = choices[0].get("text", "")
        if text:
            return text

    # Fallback: look for common fields
    for key in ("content", "text", "output", "result", "completion"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val

    return ""
