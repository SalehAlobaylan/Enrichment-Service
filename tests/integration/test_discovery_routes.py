from unittest.mock import patch

from fastapi.testclient import TestClient

DISCOVERY_ROUTES = (
    ("/v1/extract/telegram", {"username": "news_arab"}),
    ("/v1/extract/twitter", {"username": "news_arab"}),
    ("/v1/extract/twitter/recommendations", {"seed": "news_arab"}),
    ("/v1/extract/youtube", {"channel": "@news"}),
    ("/v1/extract/youtube/related", {"channel": "@news"}),
    ("/v1/extract/youtube/search", {"query": "news"}),
    ("/v1/extract/youtube/podcast-search", {"query": "podcast"}),
    ("/v1/extract/youtube/parse-feed", {"raw": {}}),
    ("/v1/extract/youtube/resolve-links", {"inputs": ["@news"]}),
    ("/v1/extract/apple-podcast/related", {"collection_id": "123"}),
)


def test_every_discovery_route_requires_the_service_bearer_token(client: TestClient) -> None:
    for path, body in DISCOVERY_ROUTES:
        response = client.post(path, json=body)
        assert response.status_code in (401, 403), path


def test_youtube_parse_feed_route_uses_real_parser_and_propagates_request_id(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    channel_id = "UC" + "f" * 22
    headers = {**auth_headers, "X-Request-ID": "discovery-fixture"}
    response = client.post(
        "/v1/extract/youtube/parse-feed",
        json={
            "raw": {
                "channelRenderer": {
                    "channelId": channel_id,
                    "title": {"simpleText": "Fixture channel"},
                }
            }
        },
        headers=headers,
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "discovery-fixture"
    assert response.json()["channels"] == [
        {
            "channel_id": channel_id,
            "title": "Fixture channel",
            "handle": None,
            "is_podcast": False,
            "episode_count": 0,
            "subscribers": 0,
            "mention_count": 1,
        }
    ]


def test_telegram_route_maps_transport_failures_to_a_safe_error_contract(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    with patch(
        "src.extraction.services.telegram_channel._get_fetcher",
        side_effect=RuntimeError("upstream body: private-token"),
    ):
        response = client.post(
            "/v1/extract/telegram",
            json={"username": "news_arab"},
            headers={**auth_headers, "X-Request-ID": "safe-error"},
        )

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "EXTRACTION_FAILED"
    assert body["request_id"] == "safe-error"
    assert "private-token" not in body["error"]
