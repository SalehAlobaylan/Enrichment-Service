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

    # Models
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
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

    # Timeouts
    TRANSCRIBE_TIMEOUT_SEC: int = 600
    EXTRACT_TIMEOUT_SEC: int = 30
    CMS_REQUEST_TIMEOUT_SEC: int = 10

    # Upload limits
    MAX_UPLOAD_MB: int = 200  # podcasts can be ~150 MB; cap above that

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

        if self.is_production and not self.service_auth_token:
            errors.append(
                "SERVICE_AUTH_TOKEN (or ENRICHMENT_SERVICE_TOKEN / "
                "CMS_SERVICE_TOKEN) must be set in production"
            )

        return errors, warnings
