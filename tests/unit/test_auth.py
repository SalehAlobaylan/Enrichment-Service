from fastapi.testclient import TestClient


def test_rejects_no_token(client: TestClient) -> None:
    resp = client.get("/v1/models")
    assert resp.status_code in (401, 403)


def test_rejects_bad_token(client: TestClient) -> None:
    resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong-token"})
    assert resp.status_code == 401


def test_accepts_valid_token(client: TestClient, auth_headers: dict[str, str]) -> None:
    resp = client.get("/v1/models", headers=auth_headers)
    assert resp.status_code == 200
