from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def test_embed_success(client: TestClient, auth_headers: dict[str, str]) -> None:
    client.app.state.model_manager.embedder.encode = MagicMock(  # type: ignore[union-attr]
        return_value=[[0.1] * 384]
    )
    resp = client.post(
        "/v1/embed",
        json={"texts": ["hello world"]},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["embeddings"]) == 1
    assert data["dimensions"] == 384
    assert data["model"] == "all-MiniLM-L6-v2"


def test_embed_query_success(client: TestClient, auth_headers: dict[str, str]) -> None:
    client.app.state.model_manager.embedder.encode = MagicMock(  # type: ignore[union-attr]
        return_value=[[0.5] * 384]
    )
    resp = client.post(
        "/v1/embed/query",
        json={"text": "search query"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["embedding"]) == 384


def test_embed_rejects_empty_texts(client: TestClient, auth_headers: dict[str, str]) -> None:
    resp = client.post(
        "/v1/embed",
        json={"texts": []},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_embed_mismatched_content_ids(client: TestClient, auth_headers: dict[str, str]) -> None:
    resp = client.post(
        "/v1/embed",
        json={"texts": ["a", "b"], "content_ids": ["id-1"]},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_embed_requires_auth(client: TestClient) -> None:
    resp = client.post("/v1/embed", json={"texts": ["hello"]})
    assert resp.status_code in (401, 403)
