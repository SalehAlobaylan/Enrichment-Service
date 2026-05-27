"""Integration tests for POST /v1/feed/news/slide — auth, validation, end-to-end shape."""
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient


def test_feed_news_slide_happy_path(client: TestClient, auth_headers: dict[str, str]) -> None:
    cms = client.app.state.cms_client  # type: ignore[union-attr]
    cms.get_content_item_basic = AsyncMock(
        return_value={
            "id": "anchor-1",
            "type": "ARTICLE",
            "title": "Saudi climate announcement",
            "excerpt": "summary",
            "source_name": "wam",
            "published_at": "2026-05-20T12:00:00Z",
        }
    )
    cms.get_content_embeddings = AsyncMock(
        return_value={
            "embedding": [0.1] * 1024,
            "embedding_sparse": {"100": 0.5},
        }
    )
    cms.knn_dense = AsyncMock(
        return_value=[{"id": "tweet-1", "type": "TWEET"}, {"id": "tweet-2", "type": "TWEET"}]
    )
    cms.knn_sparse = AsyncMock(return_value=[])
    cms.batch_text = AsyncMock(
        return_value=[
            {
                "id": "anchor-1",
                "type": "ARTICLE",
                "title": "Saudi climate announcement",
                "excerpt": "summary",
                "body_text": None,
                "source_name": "wam",
                "published_at": "2026-05-20T12:00:00Z",
            },
            {
                "id": "tweet-1",
                "type": "TWEET",
                "title": None,
                "excerpt": None,
                "body_text": "Great announcement!",
                "source_name": "twitter-user-a",
                "published_at": "2026-05-20T13:00:00Z",
            },
            {
                "id": "tweet-2",
                "type": "TWEET",
                "title": None,
                "excerpt": None,
                "body_text": "Another perspective",
                "source_name": "twitter-user-b",
                "published_at": "2026-05-20T14:00:00Z",
            },
        ]
    )
    # Reranker returns descending scores so order = batch order (preserving
    # the test's expected ordering).
    client.app.state.model_manager.reranker.rerank.return_value = [0.9, 0.7]  # type: ignore[union-attr]

    resp = client.post(
        "/v1/feed/news/slide",
        json={"anchor_content_id": "anchor-1", "k": 2},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["anchor"]["content_id"] == "anchor-1"
    assert data["anchor"]["title"] == "Saudi climate announcement"
    assert len(data["related"]) == 2
    assert {r["content_id"] for r in data["related"]} == {"tweet-1", "tweet-2"}


def test_feed_news_slide_requires_anchor(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    resp = client.post("/v1/feed/news/slide", json={"k": 3}, headers=auth_headers)
    assert resp.status_code == 422


def test_feed_news_slide_requires_auth(client: TestClient) -> None:
    resp = client.post(
        "/v1/feed/news/slide", json={"anchor_content_id": "x"}
    )
    assert resp.status_code in (401, 403)


def test_feed_news_slide_rejects_bad_k(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    resp = client.post(
        "/v1/feed/news/slide",
        json={"anchor_content_id": "x", "k": 0},  # below min
        headers=auth_headers,
    )
    assert resp.status_code == 422
