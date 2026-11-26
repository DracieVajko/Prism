import pytest
from pathlib import Path
from unified.live_session import (
    SOURCES, LiveSessionStore, UnknownSession, BadSource, SessionClosed,
)

@pytest.fixture
def store(tmp_path):
    return LiveSessionStore(tmp_path)

def test_open_creates_two_files(store):
    s = store.create(language="sk", model="turbo", label="meeting")
    assert set(s["sources"]) == set(SOURCES)
    for source in SOURCES:
        p = Path(store.status(s["id"])["descriptors"][source]["path"])
        assert p.exists()
        assert p.read_text(encoding="utf-8").startswith(
            "# Live transcript - source: " + source + " - language: sk")
        assert p.parent.name == "text"

def test_append_writes_separate_timestamped_lines(store):
    s = store.create(language="en")
    store.append(s["id"], "mic", "hello", "item_1", 5000)
    store.append(s["id"], "device", "world", "item_2", 12000)
    mic = Path(store.status(s["id"])["descriptors"]["mic"]["path"]).read_text(encoding="utf-8")
    dev = Path(store.status(s["id"])["descriptors"]["device"]["path"]).read_text(encoding="utf-8")
    assert "[00:05] hello" in mic and "world" not in mic
    assert "[00:12] world" in dev and "hello" not in dev

def test_empty_text_writes_nothing(store):
    s = store.create()
    r = store.append(s["id"], "mic", "   ", "item_empty", 0)
    assert r["bytes"] == 0
    assert len(Path(store.status(s["id"])["descriptors"]["mic"]["path"]).read_text(encoding="utf-8").splitlines()) == 1

def test_duplicate_item_id_is_ignored(store):
    s = store.create()
    store.append(s["id"], "mic", "hello", "item_1", 1000)
    r = store.append(s["id"], "mic", "hello again", "item_1", 2000)
    assert r["duplicate"] is True
    text = Path(store.status(s["id"])["descriptors"]["mic"]["path"]).read_text(encoding="utf-8")
    assert "hello again" not in text

def test_bad_source_and_unknown_session(store):
    s = store.create()
    with pytest.raises(BadSource):
        store.append(s["id"], "screen", "x", "i", 0)
    with pytest.raises(UnknownSession):
        store.append("missing", "mic", "x", "i", 0)

def test_close_is_idempotent_and_then_closed(store):
    s = store.create()
    assert store.close(s["id"])["ok"] is True
    assert store.close(s["id"])["ok"] is True
    with pytest.raises(SessionClosed):
        store.append(s["id"], "mic", "x", "i", 0)

def test_sweep_skips_open_sockets(store):
    s = store.create()
    store.mark_socket(s["id"], 1)
    store.last_activity(s["id"], 0)  # simulate 30 minutes of silence
    assert store.sweep_idle(max_idle_s=1800) == 0
    store.mark_socket(s["id"], -1)
    assert store.sweep_idle(max_idle_s=1800) == 1
