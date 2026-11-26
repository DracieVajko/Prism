"""Regression tests for the secrets store.

Covers a NUL byte reaching `os.environ`: it used to raise ValueError from
`os.environ[...] = value`, which surfaced as a 500 on `PUT /v1/settings/secrets`
in the Schemathesis fuzz step (ST101 Server error).
"""
from __future__ import annotations

import pytest

from unified import secrets


@pytest.fixture()
def settings_file(tmp_path, monkeypatch):
    """Point the store at a temp file and restore the env var afterwards."""
    path = tmp_path / "settings.local.json"
    monkeypatch.setattr(secrets, "SETTINGS_PATH", path)
    monkeypatch.setenv("LENSWRIGHT_LLM_PROVIDER", "")
    return path


def test_save_strips_nul_byte(settings_file):
    saved = secrets.save({"llm": {"provider": "a\x00b"}})

    assert saved["llm"]["provider"] == "ab"
    assert "\x00" not in settings_file.read_text(encoding="utf-8")
    assert secrets.os.environ["LENSWRIGHT_LLM_PROVIDER"] == "ab"


def test_save_strips_nul_from_secret_field(settings_file):
    saved = secrets.save({"llm": {"api_key": "sec\x00ret"}})

    assert saved["llm"]["api_key"] == "****cret"
    assert "\x00" not in settings_file.read_text(encoding="utf-8")


def test_apply_env_drops_nul_from_hand_edited_file(settings_file, monkeypatch):
    settings_file.write_text('{"llm": {"model": "x\\u0000y"}}', encoding="utf-8")
    monkeypatch.setenv("LENSWRIGHT_LLM_MODEL", "")

    secrets.load_and_apply()

    assert secrets.os.environ["LENSWRIGHT_LLM_MODEL"] == "xy"
