from src.common.config import Settings


def test_defaults() -> None:
    s = Settings(
        SERVICE_AUTH_TOKEN="tok",
        CMS_SERVICE_TOKEN="cms-tok",
        CMS_BASE_URL="http://localhost:8080",
    )
    # Whisper + TRANSCRIBE_TIMEOUT_SEC moved to Media-Service.
    # Embedder swapped to Qwen3-Embedding-0.6B (dense-only) for Arabic retrieval.
    assert s.PORT == 5050
    assert s.EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B"
    assert s.CB_FAILURE_THRESHOLD == 5
    assert s.CB_RESET_TIMEOUT_SEC == 30
    assert s.EXTRACT_TIMEOUT_SEC == 30

    # LLM defaults: Gemini 3.5 Flash primary → DeepSeek fallback.
    # If this changes, update .env.example + skills/cranl-deploy/SKILL.md
    # + the LLM section in Enrichment-Service/CLAUDE.md to match.
    assert s.LLM_PROVIDER == "gemini"
    assert s.LLM_MODEL == "gemini-3.5-flash"
    assert s.LLM_FALLBACK_PROVIDERS == "deepseek"
    assert s.fallback_providers == ["deepseek"]
    # DeepSeek has its own model default because the primary LLM_MODEL is a
    # Gemini name that DeepSeek wouldn't accept.
    assert s.model_for("deepseek") == "deepseek-chat"
    assert s.model_for("gemini") == "gemini-3.5-flash"


def test_deepseek_provider_validation() -> None:
    """deepseek is a first-class supported provider — same validation as the
    other three. Missing key in prod is fatal; in dev it's a warning."""
    # Dev mode — missing DEEPSEEK_API_KEY downgrades to warning.
    # Clear LLM_FALLBACK_PROVIDERS so the default "deepseek" doesn't conflict
    # with primary=deepseek (which would be a fatal config error).
    s = Settings(
        ENV="development",
        LLM_PROVIDER="deepseek",
        LLM_FALLBACK_PROVIDERS="",
        SERVICE_AUTH_TOKEN="t",
        CMS_SERVICE_TOKEN="c",
        CMS_BASE_URL="x",
    )
    errors, warnings = s.validate_startup()
    assert errors == []
    assert any("DEEPSEEK_API_KEY" in w for w in warnings)

    # Prod mode — missing key is fatal.
    sp = Settings(
        ENV="production",
        LLM_PROVIDER="deepseek",
        LLM_FALLBACK_PROVIDERS="",
        SERVICE_AUTH_TOKEN="t",
        CMS_SERVICE_TOKEN="c",
        CMS_BASE_URL="x",
    )
    errors_p, _ = sp.validate_startup()
    assert any("DEEPSEEK_API_KEY" in e for e in errors_p)


def test_unknown_provider_rejected() -> None:
    """Typos in LLM_PROVIDER must fail loudly — silent fallback would mean
    the LLM surface stops working with no obvious signal."""
    s = Settings(
        LLM_PROVIDER="gemeni",  # typo
        SERVICE_AUTH_TOKEN="t",
        CMS_SERVICE_TOKEN="c",
        CMS_BASE_URL="x",
    )
    errors, _ = s.validate_startup()
    # Typo path produces a fatal error regardless of ENV.
    assert any("gemeni" in e and "not supported" in e for e in errors)


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


def test_production_requires_dedicated_restart_capability() -> None:
    settings = Settings(
        ENV="production",
        SERVICE_AUTH_TOKEN="service-token",
        CMS_SERVICE_TOKEN="cms-token",
        CMS_BASE_URL="http://cms.test",
        LLM_FALLBACK_PROVIDERS="",
    )
    errors, _ = settings.validate_startup()
    assert "ENRICHMENT_RESTART_TOKEN must be set in production" in errors


def test_fallback_model_uses_its_own_provider_default() -> None:
    settings = Settings(
        LLM_PROVIDER="gemini",
        LLM_MODEL="gemini-3.5-flash",
        LLM_FALLBACK_PROVIDERS="openai,anthropic,deepseek",
    )
    assert settings.model_for("gemini") == "gemini-3.5-flash"
    assert settings.model_for("openai") == "gpt-4o-mini"
    assert settings.model_for("anthropic") == "claude-haiku-4-5-20251001"
    assert settings.model_for("deepseek") == "deepseek-chat"


def test_duplicate_fallback_provider_is_rejected() -> None:
    settings = Settings(LLM_PROVIDER="none", LLM_FALLBACK_PROVIDERS="openai,openai")
    errors, _ = settings.validate_startup()
    assert any("duplicate provider" in error for error in errors)


def test_role_and_remote_reranker_credentials_are_validated() -> None:
    invalid_role = Settings(ENRICHMENT_ROLE="worker")
    errors, _ = invalid_role.validate_startup()
    assert "ENRICHMENT_ROLE must be one of: api, reranker" in errors

    remote = Settings(
        ENV="production",
        SERVICE_AUTH_TOKEN="inbound",
        ENRICHMENT_RESTART_TOKEN="restart",
        RERANKER_BASE_URL="https://reranker.internal",
        LLM_FALLBACK_PROVIDERS="",
    )
    errors, _ = remote.validate_startup()
    assert "RERANKER_SERVICE_TOKEN must be set for a production remote reranker" in errors


def test_production_does_not_accept_cms_credential_as_inbound_token() -> None:
    settings = Settings(
        ENV="production",
        CMS_SERVICE_TOKEN="cms-only",
        ENRICHMENT_RESTART_TOKEN="restart",
        LLM_FALLBACK_PROVIDERS="",
    )
    errors, _ = settings.validate_startup()
    assert settings.service_auth_token == ""
    assert "SERVICE_AUTH_TOKEN must be set in production for inbound requests" in errors


def test_reranker_role_does_not_require_unused_llm_configuration() -> None:
    settings = Settings(
        ENV="production",
        ENRICHMENT_ROLE="reranker",
        SERVICE_AUTH_TOKEN="reranker-inbound",
        ENRICHMENT_RESTART_TOKEN="restart",
    )
    errors, _ = settings.validate_startup()
    assert not any("LLM_" in error or "API_KEY" in error for error in errors)
