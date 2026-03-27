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
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # Timeouts
    TRANSCRIBE_TIMEOUT_SEC: int = 600
    EXTRACT_TIMEOUT_SEC: int = 30
    CMS_REQUEST_TIMEOUT_SEC: int = 10

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
