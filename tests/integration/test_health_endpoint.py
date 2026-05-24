from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert data["version"] == "1.0.0"


def test_health_no_auth_required(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200


def test_ready_when_all_loaded(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["models"]["embedder"] is True
    assert data["dependencies"]["cms"] is True


def test_ready_returns_503_when_not_ready(client: TestClient) -> None:
    client.app.state.model_manager.all_ready = False  # type: ignore[union-attr]
    client.app.state.model_manager.is_ready = {"embedder": False}  # type: ignore[union-attr]
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "10"
    data = resp.json()
    assert data["status"] == "not_ready"
