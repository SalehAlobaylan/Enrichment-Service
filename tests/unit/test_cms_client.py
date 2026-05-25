import respx
from httpx import Response

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
async def test_health_check_uses_public_cms_health_endpoint() -> None:
    client = CMSClient(make_settings("http://localhost:8080/internal"))
    route = respx.get("http://localhost:8080/health").mock(
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
    route = respx.get("http://localhost:8080/health").mock(
        return_value=Response(200, json={"status": "ok"})
    )

    assert await client.health_check() is True
    assert route.called

    await client.close()
