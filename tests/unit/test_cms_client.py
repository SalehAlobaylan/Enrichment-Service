import json

import httpx
import pytest
import respx
from httpx import Response

from src.common.clients.circuit_breaker import CircuitState
from src.common.clients.cms import CMSClient
from src.common.config import Settings


def make_settings(cms_base_url: str) -> Settings:
    return Settings(
        CMS_BASE_URL=cms_base_url,
        CMS_SERVICE_TOKEN="cms-token",
        SERVICE_AUTH_TOKEN="enrichment-token",
    )


def test_build_url_avoids_duplicate_internal_prefix() -> None:
    client = CMSClient(make_settings("http://localhost:8080/internal"))
    assert (
        client._build_url("/internal/transcripts")
        == "http://localhost:8080/internal/transcripts"
    )


@respx.mock
async def test_health_check_uses_public_cms_liveness_endpoint() -> None:
    client = CMSClient(make_settings("http://localhost:8080/internal"))
    route = respx.get("http://localhost:8080/live").mock(
        return_value=Response(200, json={"status": "ok"})
    )

    assert await client.health_check() is True
    assert route.called

    await client.close()


@respx.mock
async def test_health_check_works_without_service_token() -> None:
    settings = Settings(
        CMS_BASE_URL="http://localhost:8080/internal",
        CMS_SERVICE_TOKEN="",
        SERVICE_AUTH_TOKEN="enrichment-token",
    )
    client = CMSClient(settings)
    route = respx.get("http://localhost:8080/live").mock(
        return_value=Response(200, json={"status": "ok"})
    )

    assert await client.health_check() is True
    assert route.called

    await client.close()


@respx.mock
async def test_ai_spend_failure_does_not_open_core_cms_breaker() -> None:
    settings = Settings(
        CMS_BASE_URL="http://localhost:8080",
        CMS_SERVICE_TOKEN="cms-token",
        SERVICE_AUTH_TOKEN="enrichment-token",
        CB_FAILURE_THRESHOLD=1,
    )
    client = CMSClient(settings)
    respx.post("http://localhost:8080/internal/ai-spend/events").mock(
        return_value=Response(503, json={"error": "unavailable"})
    )
    respx.put("http://localhost:8080/internal/content-items/item-1").mock(
        return_value=Response(200, json={"ok": True})
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client.emit_ai_spend_events([{"provider": "test"}])

    assert client.telemetry_circuit_breaker.state == CircuitState.OPEN
    assert client.circuit_breaker.state == CircuitState.CLOSED
    assert await client.update_content("item-1", {"title": "still works"}) == {"ok": True}

    await client.close()


def test_production_cms_client_uses_only_the_outbound_cms_token() -> None:
    client = CMSClient(
        Settings(
            ENV="production",
            SERVICE_AUTH_TOKEN="inbound-token",
            CMS_ENRICHMENT_SERVICE_TOKEN="cms-enrichment-token",
        )
    )
    assert client.token == "cms-enrichment-token"


@respx.mock
async def test_knn_dense_sends_kind_and_format_filters_independently() -> None:
    client = CMSClient(make_settings("http://localhost:8080"))
    route = respx.post("http://localhost:8080/internal/content-items/knn").mock(
        return_value=Response(200, json={"hits": []})
    )

    assert await client.knn_dense(
        [0.1, 0.2],
        "qwen-v1",
        types=["NEWS"],
        formats=["ARTICLE", "TWEET"],
        k=8,
        exclude_ids=["anchor"],
    ) == []
    assert route.called
    assert json.loads(route.calls[0].request.content) == {
        "embedding": [0.1, 0.2],
        "space_id": "qwen-v1",
        "types": ["NEWS"],
        "formats": ["ARTICLE", "TWEET"],
        "k": 8,
        "exclude_ids": ["anchor"],
    }

    await client.close()
