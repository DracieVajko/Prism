"""Local AI monitor — loaded-model detection, slot acquire/wait, unload."""
import pytest
from unified import llm_monitor


class _Resp:
    def __init__(self, payload=None, status=200):
        self._payload = payload or {}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _Client:
    """Minimal httpx.Client stand-in that records calls."""

    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers=None):
        self.state["calls"].append(("GET", url, None))
        handler = self.state.get("get")
        return handler(url) if handler else _Resp()

    def post(self, url, json=None, headers=None):
        self.state["calls"].append(("POST", url, json))
        handler = self.state.get("post")
        return handler(url, json) if handler else _Resp()

    def close(self):
        pass


class _FakeClock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture(autouse=True)
def provider_env(monkeypatch):
    monkeypatch.setenv("PRISM_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("PRISM_LLM_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("PRISM_LLM_API_KEY", "")
    monkeypatch.setenv("PRISM_LLM_MODEL", "gemma4:12b")
    monkeypatch.setattr(llm_monitor, "MAX_CONCURRENT", 1)
    monkeypatch.setattr(llm_monitor, "POLL_INTERVAL", 120)
    monkeypatch.setattr(llm_monitor, "WAIT_TIMEOUT", 3600)


@pytest.fixture
def http(monkeypatch):
    state = {"calls": []}
    monkeypatch.setattr(llm_monitor.httpx, "Client", lambda *a, **k: _Client(state))
    return state


def _models_backend(http, sequence):
    """Serve /api/ps from a list of model-name lists, last one repeating."""
    remaining = list(sequence)

    def handler(url):
        names = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return _Resp({"models": [{"name": n} for n in names]})

    http["get"] = handler


def test_loaded_models_disabled_provider(monkeypatch, http):
    monkeypatch.setenv("PRISM_LLM_PROVIDER", "none")
    assert llm_monitor.loaded_models() == []
    assert http["calls"] == []


def test_loaded_models_without_base_url(monkeypatch, http):
    monkeypatch.setenv("PRISM_LLM_BASE_URL", "")
    assert llm_monitor.loaded_models() == []
    assert http["calls"] == []


def test_loaded_models_ollama(http):
    _models_backend(http, [["a:1b", "b:2b"]])
    assert llm_monitor.loaded_models() == ["a:1b", "b:2b"]
    assert http["calls"][0][1].endswith("/api/ps")


def test_loaded_models_ollama_accepts_model_key(http):
    http["get"] = lambda url: _Resp({"models": [{"model": "m1"}, {}]})
    assert llm_monitor.loaded_models() == ["m1"]


def test_loaded_models_lmstudio(monkeypatch, http):
    monkeypatch.setenv("PRISM_LLM_PROVIDER", "lmstudio")
    http["get"] = lambda url: _Resp({"data": [{"id": "qwen"}, {"id": ""}]})
    assert llm_monitor.loaded_models() == ["qwen"]


def test_loaded_models_swallows_http_errors(http):
    def boom(url):
        raise RuntimeError("connection refused")

    http["get"] = boom
    assert llm_monitor.loaded_models() == []


def test_acquire_returns_true_without_provider(monkeypatch, http):
    monkeypatch.setenv("PRISM_LLM_PROVIDER", "none")
    assert llm_monitor.acquire("gemma4:12b") is True
    assert http["calls"] == []


def test_acquire_reuses_already_loaded_model(http):
    _models_backend(http, [["gemma4:12b"]])
    assert llm_monitor.acquire("gemma4:12b") is True
    assert [c[0] for c in http["calls"]] == ["GET"]


def test_acquire_preloads_when_slot_is_free(http):
    _models_backend(http, [[]])
    assert llm_monitor.acquire("gemma4:12b") is True
    posts = [c for c in http["calls"] if c[0] == "POST"]
    assert len(posts) == 1
    assert posts[0][1].endswith("/api/generate")
    assert posts[0][2]["model"] == "gemma4:12b"
    assert posts[0][2]["keep_alive"] == "10m"


def test_acquire_waits_then_succeeds(monkeypatch, http):
    clock = _FakeClock()
    monkeypatch.setattr(llm_monitor, "time", clock)
    _models_backend(http, [["other:7b"], []])
    assert llm_monitor.acquire("gemma4:12b", timeout=600, poll_interval=120) is True
    assert clock.now == 1000.0 + 120
    assert any(c[1].endswith("/api/generate") for c in http["calls"])


def test_acquire_returns_false_after_timeout(monkeypatch, http):
    clock = _FakeClock()
    monkeypatch.setattr(llm_monitor, "time", clock)
    _models_backend(http, [["other:7b"]])
    assert llm_monitor.acquire("gemma4:12b", timeout=300, poll_interval=120) is False
    assert clock.now >= 1000.0 + 300
    assert not any(c[0] == "POST" for c in http["calls"])


def test_acquire_allows_configured_concurrency(monkeypatch, http):
    monkeypatch.setattr(llm_monitor, "MAX_CONCURRENT", 2)
    _models_backend(http, [["other:7b"]])
    assert llm_monitor.acquire("gemma4:12b") is True
    assert any(c[0] == "POST" for c in http["calls"])


def test_release_unloads_ollama_model(http):
    assert llm_monitor.release("gemma4:12b") is None
    posts = [c for c in http["calls"] if c[0] == "POST"]
    assert len(posts) == 1
    assert posts[0][2]["keep_alive"] == 0


def test_release_unloads_lmstudio_model(monkeypatch, http):
    monkeypatch.setenv("PRISM_LLM_PROVIDER", "lmstudio")
    llm_monitor.release("qwen")
    posts = [c for c in http["calls"] if c[0] == "POST"]
    assert len(posts) == 1
    assert posts[0][1].endswith("/api/v0/models/unload")


def test_release_ignores_generic_provider(monkeypatch, http):
    monkeypatch.setenv("PRISM_LLM_PROVIDER", "openai")
    llm_monitor.release("gpt-4o-mini")
    assert http["calls"] == []


def test_llm_slot_releases_on_exit(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_monitor, "acquire", lambda model=None: calls.append("acquire") or True)
    monkeypatch.setattr(llm_monitor, "release", lambda model=None: calls.append("release"))
    with llm_monitor.LLMSlot("gemma4:12b") as slot:
        assert slot.acquired is True
    assert calls == ["acquire", "release"]


def test_llm_slot_skips_release_when_not_acquired(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_monitor, "acquire", lambda model=None: False)
    monkeypatch.setattr(llm_monitor, "release", lambda model=None: calls.append("release"))
    with llm_monitor.LLMSlot("gemma4:12b") as slot:
        assert slot.acquired is False
    assert calls == []
