"""Registry defaults and models.json fallback."""
import json

import unified.registry as reg


def test_build_registry_has_both_families(monkeypatch, tmp_path):
    monkeypatch.setattr(reg, "HERE", tmp_path)
    built = reg.build_registry()
    assert built["port"] == 8101
    assert built["default_image_model"] == "DreamShaper_8"
    assert "turbo" in built["audio_models"]
    assert built["audio_info"]["large"]["params"] == "1550M"


def test_load_image_registry_reads_models_json(monkeypatch, tmp_path):
    monkeypatch.setattr(reg, "HERE", tmp_path)
    (tmp_path / "models.json").write_text(json.dumps({
        "port": 9000,
        "default_model": "TestModel",
        "models": {"TestModel": {"path": "x"}},
        "negative_prompt": "bad",
    }), encoding="utf-8")
    img = reg.load_image_registry()
    assert img["port"] == 9000
    assert img["default_model"] == "TestModel"
    assert img["models"]["TestModel"]["path"] == "x"
