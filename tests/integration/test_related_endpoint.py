"""Integration tests for POST /v1/related — auth, validation, end-to-end shape."""
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient


def test_related_text_path(client: TestClient, auth_headers: dict[str, str]) -> None:
    cms = client.app.state.cms_client  # AsyncMock from conftest  # type: ignore[union-attr]
    cms.knn_dense = AsyncMock(
        return_value=[{"id": "a", "type": "NEWS", "format": "TWEET"}]
    )

    resp = client.post(
        "/v1/related",
        json={
            "text": "Saudi climate initiative",
            "types": ["NEWS"],
            "formats": ["TWEET"],
            "k": 5,
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["content_id"] == "a"
    assert data["results"][0]["sources"] == ["dense"]


def test_related_content_id_path(client: TestClient, auth_headers: dict[str, str]) -> None:
    cms = client.app.state.cms_client  # type: ignore[union-attr]
    cms.get_content_embeddings = AsyncMock(
        return_value={
            "embedding": [0.1] * 1024,
            "embedding_space_id": "qwen-test-space-v1",
        }
    )
    cms.knn_dense = AsyncMock(
        return_value=[{"id": "b", "type": "NEWS", "format": "COMMENT"}]
    )

    resp = client.post(
        "/v1/related",
        json={"content_id": "anchor-1", "k": 3},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"][0]["content_id"] == "b"
    assert data["results"][0]["sources"] == ["dense"]


def test_related_rejects_missing_anchor(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    resp = client.post("/v1/related", json={"k": 3}, headers=auth_headers)
    assert resp.status_code == 422  # pydantic validator


def test_related_rejects_both_anchors(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    resp = client.post(
        "/v1/related",
        json={"content_id": "a", "text": "b", "k": 3},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_related_requires_auth(client: TestClient) -> None:
    resp = client.post("/v1/related", json={"text": "hello"})
    assert resp.status_code in (401, 403)
