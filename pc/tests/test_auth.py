"""Optional bearer-token gate for the PC API."""
from fastapi.testclient import TestClient

import unified_server as us


def test_health_stays_open_and_v1_requires_token(monkeypatch):
    monkeypatch.setattr(us, "API_TOKEN", "test-secret")
    c = TestClient(us.app)
    assert c.get("/health").status_code == 200
    assert c.get("/v1/models").status_code == 401
    r = c.get("/v1/models", headers={"Authorization": "Bearer test-secret"})
    assert r.status_code == 200


def test_wrong_token_is_401(monkeypatch):
    monkeypatch.setattr(us, "API_TOKEN", "test-secret")
    c = TestClient(us.app)
    r = c.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized"
