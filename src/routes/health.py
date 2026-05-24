from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request, Response

from src.auth.service_auth import verify_service_token
from src.schemas.common import HealthResponse, ModelInfoItem, ModelsResponse, ReadyResponse

router = APIRouter()

VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=VERSION,
    )


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request, response: Response) -> ReadyResponse:
    model_manager = request.app.state.model_manager
    cms_client = request.app.state.cms_client

    models_status = model_manager.is_ready
    cms_reachable = await cms_client.health_check()

    all_ready = model_manager.all_ready and cms_reachable

    if not all_ready:
        response.status_code = 503
        response.headers["Retry-After"] = "10"

    return ReadyResponse(
        status="ok" if all_ready else "not_ready",
        models=models_status,
        dependencies={"cms": cms_reachable},
    )


@router.get(
    "/v1/models",
    response_model=ModelsResponse,
    dependencies=[Depends(verify_service_token)],
)
async def models(request: Request) -> ModelsResponse:
    """Enrichment-Service model registry.

    Whisper + CLIP are reported by Media-Service's /v1/models endpoint.
    Slice 0 will add the BGE-M3 reranker here alongside the embedder.
    """
    model_manager = request.app.state.model_manager

    items = [
        ModelInfoItem(
            name=model_manager.embedder.model_name,
            loaded=model_manager.embedder.is_loaded,
            type="sentence-transformer",
            dimensions=model_manager.embedder.dimensions
            if model_manager.embedder.is_loaded
            else None,
        ),
    ]

    return ModelsResponse(models=items)
