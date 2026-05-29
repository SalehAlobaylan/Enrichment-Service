from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    error_code: str
    retryable: bool
    retry_after_seconds: int | None = None
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class ModelInfoItem(BaseModel):
    name: str
    loaded: bool
    type: str
    dimensions: int | None = None


class ReadyResponse(BaseModel):
    status: str
    models: dict[str, bool]
    dependencies: dict[str, bool]
    # Richer per-model identity (name + dims), additive to the bool `models`
    # map. `type` is the role (embedder/reranker) — the admin dashboard joins
    # on it to show model names/dims instead of bare load booleans.
    models_detail: list[ModelInfoItem] = []


class ModelsResponse(BaseModel):
    models: list[ModelInfoItem]
