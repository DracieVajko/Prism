"""Local AI resource monitor for external LLM providers.

Before a Prism job asks an external LLM (Ollama, LM Studio, ...) to do work,
this module checks how many models are currently resident on that provider and
decides whether we may load our needed model:

* If fewer than ``PRISM_LLM_MAX_CONCURRENT`` (default 1) models are loaded,
  we are free to load/preload our model.
* Otherwise we wait and re-check every ``PRISM_LLM_POLL_INTERVAL`` seconds
  (default 120) for up to ``PRISM_LLM_WAIT_TIMEOUT`` seconds (default 3600).
  Only after the timeout do we fall back (caller handles fallback).
* When the job is finished we release/unload the model we loaded, so the
  provider is left clean.

Provider support is best-effort:
* Ollama: full (``/api/ps`` for loaded list, ``keep_alive=0`` to unload).
* LM Studio: detection via ``/v1/models``; unload attempted via the v0 API.
* Others (OpenAI/Anthropic/custom): assumed to have no locally resident
  model limit; always allowed, no unload.
"""
import json
import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger("llm_monitor")

MAX_CONCURRENT = int(os.environ.get("PRISM_LLM_MAX_CONCURRENT", "1") or 1)
POLL_INTERVAL = int(os.environ.get("PRISM_LLM_POLL_INTERVAL", "120") or 120)
WAIT_TIMEOUT = int(os.environ.get("PRISM_LLM_WAIT_TIMEOUT", "3600") or 3600)


def _cfg() -> dict:
    return {
        "provider": os.environ.get("PRISM_LLM_PROVIDER", "none").strip().lower(),
        "base_url": os.environ.get("PRISM_LLM_BASE_URL", "").strip().rstrip("/"),
        "api_key": os.environ.get("PRISM_LLM_API_KEY", "").strip(),
        "model": os.environ.get("PRISM_LLM_MODEL", "gemma4:12b").strip(),
    }


def _headers(cfg: dict) -> dict:
    h = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        h["Authorization"] = "Bearer " + cfg["api_key"]
    return h


def loaded_models() -> list:
    """Return list of currently loaded model names on the configured provider."""
    cfg = _cfg()
    provider = cfg["provider"]
    base = cfg["base_url"]
    if not base or provider in ("none", "", "disabled"):
        return []
    try:
        if provider == "ollama":
            with httpx.Client(timeout=10.0) as c:
                r = c.get(base + "/api/ps", headers=_headers(cfg))
                r.raise_for_status()
                data = r.json()
            out = []
            for m in (data.get("models") or []):
                name = m.get("name") or m.get("model")
                if name:
                    out.append(name)
            return out
        if provider == "lmstudio":
            with httpx.Client(timeout=10.0) as c:
                r = c.get(base + "/v1/models", headers=_headers(cfg))
                r.raise_for_status()
                data = r.json()
            out = []
            for m in (data.get("data") or []):
                mid = m.get("id")
                if mid:
                    out.append(mid)
            return out
    except Exception as e:
        logger.warning("loaded_models check failed: %s", str(e)[:200])
    return []


def _load_model_ollama(base: str, model: str, headers: dict, keep_alive: str = "10m") -> bool:
    """Preload/warm a model in Ollama by issuing a no-op generate with keep_alive."""
    try:
        with httpx.Client(timeout=120.0) as c:
            r = c.post(base + "/api/generate", json={
                "model": model, "prompt": "", "keep_alive": keep_alive, "stream": False,
            }, headers=headers)
            r.raise_for_status()
        return True
    except Exception as e:
        logger.warning("ollama preload failed: %s", str(e)[:200])
        return False


def _unload_model_ollama(base: str, model: str, headers: dict) -> bool:
    try:
        with httpx.Client(timeout=30.0) as c:
            r = c.post(base + "/api/generate", json={
                "model": model, "prompt": "", "keep_alive": 0, "stream": False,
            }, headers=headers)
            r.raise_for_status()
        return True
    except Exception as e:
        logger.warning("ollama unload failed: %s", str(e)[:200])
        return False


def _unload_model_lmstudio(base: str, model: str, headers: dict) -> bool:
    # LM Studio v0 unload endpoint (best-effort).
    try:
        with httpx.Client(timeout=30.0) as c:
            r = c.post(base + "/api/v0/models/unload", json={"model": model}, headers=headers)
            if r.status_code < 400:
                return True
    except Exception as e:
        logger.warning("lmstudio unload failed: %s", str(e)[:200])
    return False


def acquire(model: Optional[str] = None, timeout: Optional[int] = None,
            poll_interval: Optional[int] = None) -> bool:
    """Wait until we are allowed to load ``model``. Returns True if acquired
    within timeout, False if timed out (caller should fall back)."""
    cfg = _cfg()
    provider = cfg["provider"]
    base = cfg["base_url"]
    if model is None:
        model = cfg["model"]
    if timeout is None:
        timeout = WAIT_TIMEOUT
    if poll_interval is None:
        poll_interval = POLL_INTERVAL
    if provider in ("none", "", "disabled") or not base:
        return True

    deadline = time.time() + timeout
    while True:
        loaded = loaded_models()
        # If our model is already loaded, we have a slot implicitly.
        if model in loaded:
            return True
        if len(loaded) < MAX_CONCURRENT:
            # Try to preload so subsequent calls are fast.
            if provider == "ollama":
                _load_model_ollama(base, model, _headers(cfg))
            return True
        if time.time() >= deadline:
            logger.warning("Timed out waiting for an LLM slot after %ds; %d models loaded",
                           timeout, len(loaded))
            return False
        logger.info("LLM capacity reached (%d/%d loaded); waiting %ds before retry",
                    len(loaded), MAX_CONCURRENT, poll_interval)
        time.sleep(poll_interval)


def release(model: Optional[str] = None) -> None:
    """Unload the model we loaded (best-effort)."""
    cfg = _cfg()
    provider = cfg["provider"]
    base = cfg["base_url"]
    if model is None:
        model = cfg["model"]
    if provider in ("none", "", "disabled") or not base:
        return
    if provider == "ollama":
        _unload_model_ollama(base, model, _headers(cfg))
    elif provider == "lmstudio":
        _unload_model_lmstudio(base, model, _headers(cfg))
    # generic providers: nothing to unload


class LLMSlot:
    """Context manager that acquires a slot on entry and releases on exit."""

    def __init__(self, model: Optional[str] = None):
        self.model = model
        self.acquired = False

    def __enter__(self):
        self.acquired = acquire(self.model)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.acquired:
            release(self.model)
        return False
