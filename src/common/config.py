from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server
    PORT: int = 5050
    ENV: str = "development"
    LOG_LEVEL: str = "info"
    WORKERS: int = 1

    # Auth
    SERVICE_AUTH_TOKEN: str = ""
    ENRICHMENT_SERVICE_TOKEN: str = ""
    CMS_SERVICE_TOKEN: str = ""
    CMS_BASE_URL: str = "http://localhost:8080"

    # Models — text embedder only. Whisper + CLIP moved to Media-Service.
    # BGE-M3 is multilingual (Arabic + English first-class), 1024-dim dense,
    # forward-compatible with sparse output for hybrid retrieval (Slice A).
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    MODELS_DIR: str = "./models"

    # Circuit Breaker
    CB_FAILURE_THRESHOLD: int = 5
    CB_RESET_TIMEOUT_SEC: int = 30
    CB_HALF_OPEN_REQUESTS: int = 3

    # LLM
    # Supported providers: "openai", "anthropic", "gemini", or "none"/"" to disable.
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    # Optional fallback providers (CSV), tried in order when the primary
    # provider exhausts its retries on a retryable error. Each named provider
    # must have its API key set.
    LLM_FALLBACK_PROVIDERS: str = ""
    # Per-provider model overrides (used when a fallback fires). If unset for
    # a given provider, falls back to provider-default sensible model.
    LLM_OPENAI_MODEL: str = ""
    LLM_ANTHROPIC_MODEL: str = ""
    LLM_GEMINI_MODEL: str = ""

    # Timeouts — TRANSCRIBE_TIMEOUT_SEC moved to Media-Service.
    EXTRACT_TIMEOUT_SEC: int = 30
    CMS_REQUEST_TIMEOUT_SEC: int = 10

    # Redis — LLM response cache. The arq queue and its DB live in
    # Media-Service now.
    REDIS_URL: str = "redis://localhost:6379"
    LLM_CACHE_DB: int = 1

    # LLM response cache.
    LLM_CACHE_ENABLED: bool = True
    LLM_CACHE_TTL_SEC: int = 604800  # 7 days

    # CORS — CSV of allowed origins. Empty string disables CORS in prod;
    # default is wide-open in dev for convenience.
    CORS_ALLOWED_ORIGINS: str = "*"

    @property
    def is_production(self) -> bool:
        return self.ENV == "production"

    @property
    def service_auth_token(self) -> str:
        return (
            self.SERVICE_AUTH_TOKEN
            or self.ENRICHMENT_SERVICE_TOKEN
            or self.CMS_SERVICE_TOKEN
        )

    @property
    def fallback_providers(self) -> list[str]:
        """Parsed LLM_FALLBACK_PROVIDERS as a clean ordered list."""
        return [
            p.strip().lower()
            for p in (self.LLM_FALLBACK_PROVIDERS or "").split(",")
            if p.strip()
        ]

    def model_for(self, provider: str) -> str:
        """Resolve which model name to use for a given provider.

        Per-provider override env vars (LLM_OPENAI_MODEL etc.) win when set;
        otherwise fall back to LLM_MODEL (which is the model name the operator
        configured for the primary provider).
        """
        override_map = {
            "openai": self.LLM_OPENAI_MODEL,
            "anthropic": self.LLM_ANTHROPIC_MODEL,
            "gemini": self.LLM_GEMINI_MODEL,
        }
        override = (override_map.get(provider) or "").strip()
        return override or self.LLM_MODEL

    def validate_startup(self) -> tuple[list[str], list[str]]:
        """Return (fatal_errors, warnings).

        Production: missing LLM keys are fatal — translate/summarize would
        crash on first call, better to fail fast at boot.
        Dev/test: same issues are downgraded to warnings so local stacks
        can boot without every operator owning an OpenAI key. The LLM
        routes will return a clear error at call-time if the key is still
        missing when used.
        """
        errors: list[str] = []
        warnings: list[str] = []
        provider = (self.LLM_PROVIDER or "").strip().lower()

        def _missing_key(msg: str) -> None:
            (errors if self.is_production else warnings).append(msg)

        if provider == "openai":
            if not self.OPENAI_API_KEY.strip():
                _missing_key("LLM_PROVIDER=openai but OPENAI_API_KEY is empty")
        elif provider == "anthropic":
            if not self.ANTHROPIC_API_KEY.strip():
                _missing_key(
                    "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty"
                )
        elif provider == "gemini":
            if not self.GEMINI_API_KEY.strip():
                _missing_key(
                    "LLM_PROVIDER=gemini but GEMINI_API_KEY is empty"
                )
        elif provider in ("", "none", "disabled"):
            pass
        else:
            # An unknown provider name is always fatal — it would silently
            # break the entire LLM surface and is almost certainly a typo.
            errors.append(
                f"LLM_PROVIDER={self.LLM_PROVIDER!r} is not supported "
                "(expected one of: openai, anthropic, gemini, none)"
            )

        # Validate fallback chain — each named provider must have its API key
        # set, and the primary should not appear in the fallback list (cycles
        # and self-references are almost certainly a misconfiguration).
        _provider_keys = {
            "openai": self.OPENAI_API_KEY,
            "anthropic": self.ANTHROPIC_API_KEY,
            "gemini": self.GEMINI_API_KEY,
        }
        for fp in self.fallback_providers:
            if fp == provider:
                errors.append(
                    f"LLM_FALLBACK_PROVIDERS contains {fp!r}, which is also the primary "
                    "provider — fallback list must exclude the primary"
                )
                continue
            if fp not in _provider_keys:
                errors.append(
                    f"LLM_FALLBACK_PROVIDERS contains unsupported provider {fp!r} "
                    "(expected: openai, anthropic, gemini)"
                )
                continue
            if not _provider_keys[fp].strip():
                _missing_key(
                    f"LLM_FALLBACK_PROVIDERS includes {fp!r} but its API key is empty"
                )

        if self.is_production and not self.service_auth_token:
            errors.append(
                "SERVICE_AUTH_TOKEN (or ENRICHMENT_SERVICE_TOKEN / "
                "CMS_SERVICE_TOKEN) must be set in production"
            )

        return errors, warnings
