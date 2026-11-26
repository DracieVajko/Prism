"""Local secrets and connection settings (HF token, LLM/API providers).

Stored in `pc/settings.local.json`, which is git-ignored. Values are applied to
`os.environ` at startup and whenever the user saves them, so every existing
consumer (`llm_providers`, `download_models`, OCR/VLM loaders) keeps reading
the same environment variables it always did.

Nothing here is ever sent anywhere except to the provider the user configured.
The GET endpoint returns secrets masked: the UI only needs to know whether a
value is set and its last four characters.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.local.json"
_LOCK = threading.Lock()

# Field name -> environment variable. Nested groups use "group.field".
_ENV_MAP = {
    "hf_token": "HF_TOKEN",
    "llm.provider": "LENSWRIGHT_LLM_PROVIDER",
    "llm.base_url": "LENSWRIGHT_LLM_BASE_URL",
    "llm.api_key": "LENSWRIGHT_LLM_API_KEY",
    "llm.model": "LENSWRIGHT_LLM_MODEL",
    "llm.endpoint": "LENSWRIGHT_LLM_ENDPOINT",
}

SECRET_FIELDS = {"hf_token", "llm.api_key"}

DEFAULTS = {
    "hf_token": "",
    "llm": {
        "provider": "none",
        "base_url": "",
        "api_key": "",
        "model": "",
        "endpoint": "",
    },
}


def _flatten(data: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in (data or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def _unflatten(flat: dict) -> dict:
    data: dict = {}
    for name, value in flat.items():
        parts = name.split(".")
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return data


def _read_file() -> dict:
    if not SETTINGS_PATH.is_file():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("settings.local.json unreadable, ignoring: %s", e)
        return {}


def _apply_env(flat: dict) -> None:
    """Push stored values into the environment (non-empty only)."""
    for name, env in _ENV_MAP.items():
        value = flat.get(name)
        if isinstance(value, str) and value.strip():
            os.environ[env] = value.strip()
    # huggingface_hub also honours this alias
    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])


def load_and_apply() -> dict:
    """Load settings from disk and apply them to the environment."""
    with _LOCK:
        flat = _flatten(_read_file())
    _apply_env(flat)
    return flat


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "****"
    return "****" + value[-4:]


def public_view() -> dict:
    """Settings with secrets masked, plus which secrets are configured."""
    flat = _flatten(_read_file())
    merged = {**_flatten(DEFAULTS), **flat}
    view = _unflatten(merged)
    for name in SECRET_FIELDS:
        parts = name.split(".")
        node = view
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        raw = flat.get(name, "")
        node[parts[-1]] = _mask(raw) if raw else ""
    view["_configured"] = {
        "hf_token": bool(flat.get("hf_token", "").strip()),
        "llm_api_key": bool(flat.get("llm.api_key", "").strip()),
    }
    return view


def save(updates: dict) -> dict:
    """Merge `updates` into the stored settings and apply them.

    A secret field sent as an empty string clears it; a field sent as the
    masked placeholder (``****abcd``) is ignored so the UI can round-trip the
    form without wiping secrets it never displayed.
    """
    with _LOCK:
        current = _flatten(_read_file())
        for name, value in _flatten(updates).items():
            if name not in _ENV_MAP:
                continue
            value = "" if value is None else str(value)
            if value.startswith("****"):
                continue
            current[name] = value.strip()
        data = _unflatten(current)
        SETTINGS_PATH.write_text(json.dumps(data, indent=4), encoding="utf-8")
        try:
            os.chmod(SETTINGS_PATH, 0o600)
        except Exception:
            pass
    _apply_env(current)
    logger.info("Settings saved (secrets redacted)")
    return public_view()


def hf_token() -> str:
    """Current HF token from env or the stored settings."""
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if token:
        return token
    return str(_flatten(_read_file()).get("hf_token", "")).strip()


def test_hf_token() -> dict:
    """Validate the HF token against the whoami endpoint."""
    token = hf_token()
    if not token:
        return {"ok": False, "detail": "No HF token configured."}
    try:
        from huggingface_hub import HfApi
        info = HfApi().whoami(token=token)
        return {"ok": True, "user": info.get("name", ""),
                "detail": f"Token valid for '{info.get('name', '?')}'."}
    except Exception as e:
        return {"ok": False, "detail": f"Token rejected: {str(e)[:200]}"}


def test_llm() -> dict:
    """Call the configured LLM with a trivial prompt."""
    from unified import llm_providers
    cfg = llm_providers._get_config()
    if cfg.get("provider") in ("", "none"):
        return {"ok": False, "detail": "No LLM provider configured."}
    try:
        text = llm_providers.process_with_external_llm(
            "Ping", instruction="Reply with OK only.")
        return {"ok": bool(text), "detail": f"Provider replied: {str(text)[:120]}"}
    except Exception as e:
        return {"ok": False, "detail": f"LLM call failed: {str(e)[:200]}"}
