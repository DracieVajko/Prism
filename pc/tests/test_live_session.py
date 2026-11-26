import pytest
from pathlib import Path
from unified.live_session import (
    SOURCES, LiveSessionStore, UnknownSession, BadSource, SessionClosed,
)

@pytest.fixture
def store(tmp_path):
    return LiveSessionStore(tmp_path)

def transcript_path(tmp_path, store, session_id, source):
    """Resolve the public relative descriptor path against the output dir."""
    descriptor = store.status(session_id)["descriptors"][source]
    assert not Path(descriptor["path"]).is_absolute()
    return tmp_path / descriptor["path"]

def test_open_creates_two_files(store, tmp_path):
    s = store.create(language="sk", model="turbo", label="meeting")
    assert set(s["sources"]) == set(SOURCES)
    assert set(s["descriptors"]) == set(SOURCES)
    assert set(store.status(s["id"])["descriptors"]) == set(SOURCES)
    for source in SOURCES:
        descriptor = s["descriptors"][source]
        assert descriptor["path"] == "text/" + descriptor["name"]
        assert descriptor["url"] == "/v1/outputs/text/" + descriptor["name"]
        p = transcript_path(tmp_path, store, s["id"], source)
        assert p.exists()
        assert p.read_text(encoding="utf-8").startswith(
            "# Live transcript - source: " + source + " - language: sk")
        assert p.parent.name == "text"

def test_append_writes_separate_timestamped_lines(store, tmp_path):
    s = store.create(language="en")
    store.append(s["id"], "mic", "hello", "item_1", 5000)
    store.append(s["id"], "device", "world", "item_2", 12000)
    mic = transcript_path(tmp_path, store, s["id"], "mic").read_text(encoding="utf-8")
    dev = transcript_path(tmp_path, store, s["id"], "device").read_text(encoding="utf-8")
    assert "[00:05] hello" in mic and "world" not in mic
    assert "[00:12] world" in dev and "hello" not in dev

def test_empty_text_writes_nothing(store, tmp_path):
    s = store.create()
    r = store.append(s["id"], "mic", "   ", "item_empty", 0)
    assert r["bytes"] == 0
    path = transcript_path(tmp_path, store, s["id"], "mic")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    # An empty append must not consume the id: a retransmit with real text
    # must still write its line.
    r = store.append(s["id"], "mic", "now with text", "item_empty", 1000)
    assert r["bytes"] > 0 and r["duplicate"] is False
    assert "[00:01] now with text" in path.read_text(encoding="utf-8")

def test_duplicate_item_id_is_ignored(store, tmp_path):
    s = store.create()
    store.append(s["id"], "mic", "hello", "item_1", 1000)
    r = store.append(s["id"], "mic", "hello again", "item_1", 2000)
    assert r["duplicate"] is True
    text = transcript_path(tmp_path, store, s["id"], "mic").read_text(encoding="utf-8")
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
