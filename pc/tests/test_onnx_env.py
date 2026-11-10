"""ONNX Runtime provider selection — override, auto order, QDQ/DML rules."""
import sys

import pytest

from unified import onnx_env


@pytest.fixture
def providers(monkeypatch):
    def _set(available):
        monkeypatch.setattr(onnx_env, "available_providers", lambda: list(available))
    return _set


def test_cpu_override_wins(monkeypatch, providers):
    monkeypatch.setenv("PRISM_ONNX_PROVIDER", "cpu")
    providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert onnx_env.pick_providers("generic") == ["CPUExecutionProvider"]


def test_cuda_override_when_available(monkeypatch, providers):
    monkeypatch.setenv("PRISM_ONNX_PROVIDER", "cuda")
    providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
    picked = onnx_env.pick_providers("generic")
    assert picked[0][0] == "CUDAExecutionProvider"
    assert picked[-1] == "CPUExecutionProvider"


def test_cuda_override_falls_back_when_missing(monkeypatch, providers):
    monkeypatch.setenv("PRISM_ONNX_PROVIDER", "cuda")
    providers(["CPUExecutionProvider"])
    assert onnx_env.pick_providers("generic") == ["CPUExecutionProvider"]


def test_dml_override_skips_qdq_models(monkeypatch, providers):
    monkeypatch.setenv("PRISM_ONNX_PROVIDER", "dml")
    providers(["DmlExecutionProvider", "CPUExecutionProvider"])
    assert onnx_env.pick_providers("qdq") == ["CPUExecutionProvider"]


def test_dml_override_used_for_generic_models(monkeypatch, providers):
    monkeypatch.setenv("PRISM_ONNX_PROVIDER", "dml")
    providers(["DmlExecutionProvider", "CPUExecutionProvider"])
    assert onnx_env.pick_providers("generic") == [
        "DmlExecutionProvider", "CPUExecutionProvider"]


def test_auto_prefers_cuda_over_dml(monkeypatch, providers):
    monkeypatch.delenv("PRISM_ONNX_PROVIDER", raising=False)
    providers(["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
    picked = onnx_env.pick_providers("generic")
    assert picked[0][0] == "CUDAExecutionProvider"


def test_auto_qdq_never_uses_dml(monkeypatch, providers):
    monkeypatch.delenv("PRISM_ONNX_PROVIDER", raising=False)
    providers(["DmlExecutionProvider", "CPUExecutionProvider"])
    assert onnx_env.pick_providers("qdq") == ["CPUExecutionProvider"]


def test_auto_generic_uses_dml_when_no_cuda(monkeypatch, providers):
    monkeypatch.delenv("PRISM_ONNX_PROVIDER", raising=False)
    providers(["DmlExecutionProvider", "CPUExecutionProvider"])
    assert onnx_env.pick_providers("generic") == [
        "DmlExecutionProvider", "CPUExecutionProvider"]


def test_auto_without_accelerators_uses_cpu(monkeypatch, providers):
    monkeypatch.delenv("PRISM_ONNX_PROVIDER", raising=False)
    providers(["CPUExecutionProvider"])
    assert onnx_env.pick_providers("generic") == ["CPUExecutionProvider"]


def test_pick_providers_survives_ort_import_failure(monkeypatch):
    monkeypatch.setattr(onnx_env, "available_providers", lambda: [])
    monkeypatch.setenv("PRISM_ONNX_PROVIDER", "auto")
    assert onnx_env.pick_providers("generic") == ["CPUExecutionProvider"]


def test_provider_label_known_and_unknown():
    assert onnx_env.provider_label([]) == "unknown"
    assert onnx_env.provider_label(["CPUExecutionProvider"]) == "cpu"
    assert onnx_env.provider_label(["DmlExecutionProvider", "CPUExecutionProvider"]) == "directml"
    assert onnx_env.provider_label([("CUDAExecutionProvider", {"x": 1})]) == "cuda"
    assert onnx_env.provider_label(["WebGpuExecutionProvider"]) == "WebGpuExecutionProvider"


def test_register_nvidia_dlls_is_noop_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert onnx_env.register_nvidia_dlls() == []
