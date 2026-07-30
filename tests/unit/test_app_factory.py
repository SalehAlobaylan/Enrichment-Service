from src.common.config import Settings
from src.main import create_app


def _paths(settings: Settings) -> set[str]:
    return {route.path for route in create_app(settings).routes}


def test_reranker_role_mounts_only_operations_and_rerank_surface() -> None:
    paths = _paths(Settings(ENRICHMENT_ROLE="reranker", CORS_ALLOWED_ORIGINS=""))
    assert "/health" in paths
    assert "/ready" in paths
    assert "/v1/rerank" in paths
    assert "/v1/embed" not in paths
    assert "/v1/extract" not in paths
    assert "/v1/translate" not in paths


def test_api_role_mounts_the_full_text_intelligence_surface() -> None:
    paths = _paths(Settings(ENRICHMENT_ROLE="api", CORS_ALLOWED_ORIGINS=""))
    assert {
        "/v1/embed",
        "/v1/related",
        "/v1/extract",
        "/v1/translate",
        "/v1/stories/digest",
        "/v1/stories/label",
    } <= paths
    assert "/v1/rerank" in paths
