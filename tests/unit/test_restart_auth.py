from fastapi.testclient import TestClient

from src.common.routes import admin


def test_restart_rejects_the_ordinary_service_token(client: TestClient) -> None:
    response = client.post(
        "/v1/admin/restart", headers={"Authorization": "Bearer test-token"}
    )
    assert response.status_code == 401


def test_restart_requires_the_dedicated_capability(client: TestClient) -> None:
    response = client.post("/v1/admin/restart")
    assert response.status_code in (401, 403)


def test_restart_accepts_dedicated_capability_without_killing_test_process(
    client: TestClient, monkeypatch
) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(admin, "terminate_process", lambda pid, sig: calls.append((pid, sig)))

    response = client.post(
        "/v1/admin/restart",
        headers={
            "Authorization": "Bearer test-restart-token",
            "X-Operator-Email": "operator@example.test",
            "X-Request-ID": "restart-test-request",
        },
    )

    assert response.status_code == 202
    assert response.json()["service"] == "enrichment"
    assert len(calls) == 1
