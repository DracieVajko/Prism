"""ONNX provider selection without loading onnxruntime."""
import unified.onnx_env as oe


def test_override_cpu(monkeypatch):
    monkeypatch.setenv("LENSWRIGHT_ONNX_PROVIDER", "cpu")
    monkeypatch.setattr(oe, "available_providers", lambda: ["CUDAExecutionProvider"])
    assert oe.pick_providers("generic") == ["CPUExecutionProvider"]


def test_qdq_skips_directml(monkeypatch):
    monkeypatch.setenv("LENSWRIGHT_ONNX_PROVIDER", "auto")
    monkeypatch.setattr(oe, "available_providers", lambda: ["DmlExecutionProvider"])
    assert oe.pick_providers("qdq") == ["CPUExecutionProvider"]
    assert oe.pick_providers("generic") == [
        "DmlExecutionProvider", "CPUExecutionProvider"]


def test_cuda_preferred(monkeypatch):
    monkeypatch.setenv("LENSWRIGHT_ONNX_PROVIDER", "auto")
    monkeypatch.setattr(oe, "available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    providers = oe.pick_providers("generic")
    assert providers[-1] == "CPUExecutionProvider"
    assert providers[0][0] == "CUDAExecutionProvider"


def test_provider_label():
    assert oe.provider_label([]) == "unknown"
    assert oe.provider_label(["CUDAExecutionProvider", "CPUExecutionProvider"]) == "cuda"
    assert oe.provider_label([("CUDAExecutionProvider", {})]) == "cuda"
