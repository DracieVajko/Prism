"""Regression checks for the CI workflow's Schemathasis invocation."""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _schemathesis_run_text() -> str:
    assert WORKFLOW.is_file(), f"missing workflow: {WORKFLOW}"
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = data.get("jobs", {})
    pc_job = jobs.get("pc", {})
    for step in pc_job.get("steps", []):
        run = step.get("run") or ""
        if "schemathesis run" in run:
            return run
    raise AssertionError("no Schemathesis step found in pc job")


def test_no_unsupported_schemathasis_option():
    run = _schemathesis_run_text()
    assert "--exit-code-on-first-failure" not in run, (
        "Schemathasis 4.x removed --exit-code-on-first-failure"
    )


def test_schemathasis_fail_fast_and_nonzero_exit():
    run = _schemathesis_run_text()
    assert "--max-failures" in run
    assert "exit $SCHEMATHESIS_EXIT" in run
    assert "schemathesis run" in run


def test_schemathasis_checks_meaningful():
    run = _schemathesis_run_text()
    assert "--checks not_a_server_error" in run
    assert "--exclude-checks" not in run