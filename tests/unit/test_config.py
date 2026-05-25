from src.common.config import Settings


def test_defaults() -> None:
    s = Settings(
        SERVICE_AUTH_TOKEN="tok",
        CMS_SERVICE_TOKEN="cms-tok",
        CMS_BASE_URL="http://localhost:8080",
    )
    # Whisper + TRANSCRIBE_TIMEOUT_SEC moved to Media-Service.
    # Embedder swapped to BGE-M3 in Slice 0 for Arabic-native retrieval.
    assert s.PORT == 5050
    assert s.EMBEDDING_MODEL == "BAAI/bge-m3"
    assert s.CB_FAILURE_THRESHOLD == 5
    assert s.CB_RESET_TIMEOUT_SEC == 30
    assert s.EXTRACT_TIMEOUT_SEC == 30


def test_is_production() -> None:
    s = Settings(
        ENV="production",
        SERVICE_AUTH_TOKEN="t",
        CMS_SERVICE_TOKEN="c",
        CMS_BASE_URL="x",
    )
    assert s.is_production is True

    s2 = Settings(
        ENV="development",
        SERVICE_AUTH_TOKEN="t",
        CMS_SERVICE_TOKEN="c",
        CMS_BASE_URL="x",
    )
    assert s2.is_production is False


def test_service_auth_token_fallbacks() -> None:
    s = Settings(
        ENRICHMENT_SERVICE_TOKEN="enrichment-token",
        CMS_SERVICE_TOKEN="cms-token",
        CMS_BASE_URL="x",
    )
    assert s.service_auth_token == "enrichment-token"

    s2 = Settings(
        CMS_SERVICE_TOKEN="cms-token",
        CMS_BASE_URL="x",
    )
    assert s2.service_auth_token == "cms-token"
