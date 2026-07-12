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
    # Immutable vector-space descriptor (stage 10 — Embedding & Model Lifecycle).
    # Present on embedder items; absent/None on reranker (stateless, no space).
    # revision == "" or space_id == "" ⇒ the space is lifecycle-not-ready.
    revision: str | None = None
    normalized: bool | None = None
    pooling: str | None = None
    space_id: str | None = None
    producer_recipe: str | None = None
    producer_id: str | None = None


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
