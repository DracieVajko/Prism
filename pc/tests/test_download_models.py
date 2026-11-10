"""Model catalog integrity and download helpers (no network)."""
import pytest

import download_models as dm


ALL_TASK_LISTS = [
    dm.TASKS,
    dm.VLM_TASKS,
    dm.OCR_TASKS,
    dm.AUDIO_TASKS,
    dm.UPSCALE_TASKS,
]


def test_every_task_has_required_fields():
    for tasks in ALL_TASK_LISTS:
        for task in tasks:
            assert task.get("id"), task
            assert task.get("target"), task
            assert isinstance(task.get("min_gb"), (int, float)), task
            kind = task.get("kind")
            assert kind in ("file", "snapshot", "url"), task
            if kind == "url":
                assert task.get("url"), task
                assert task.get("file"), task
            else:
                assert task.get("repo"), task
                if kind == "file":
                    assert task.get("file"), task


def test_task_ids_are_unique_across_catalogs():
    ids = [t["id"] for tasks in ALL_TASK_LISTS for t in tasks]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert duplicates == set()


def test_presets_reference_known_tasks():
    known = {t["id"] for tasks in ALL_TASK_LISTS for t in tasks}
    known |= set(dm.PRESETS.get("odysseus-default", []))
    for name, ids in dm.PRESETS.items():
        assert ids, f"preset {name} is empty"
        unknown = [i for i in ids if i not in known]
        assert unknown == [], f"preset {name} references {unknown}"


def test_odysseus_default_covers_all_sd_tasks():
    assert dm.PRESETS["odysseus-default"] == [t["id"] for t in dm.TASKS]


def test_dir_size_gb_sums_files(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 500)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"x" * 500)
    assert dm.dir_size_gb(tmp_path) == pytest.approx(1000 / 1e9)


def test_dir_size_gb_ignores_missing_tree(tmp_path):
    assert dm.dir_size_gb(tmp_path / "nope") == 0.0


def test_run_with_retries_returns_first_success():
    assert dm.run_with_retries(lambda: 42, "ok", tries=3) == 42


def test_run_with_retries_retries_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(dm.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("network down")
        return "done"

    assert dm.run_with_retries(flaky, "flaky", tries=3) == "done"
    assert sleeps == [10, 20]


def test_run_with_retries_raises_after_last_attempt(monkeypatch):
    monkeypatch.setattr(dm.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise OSError("nope")

    with pytest.raises(OSError):
        dm.run_with_retries(always_fails, "bad", tries=2)
    assert calls["n"] == 2
