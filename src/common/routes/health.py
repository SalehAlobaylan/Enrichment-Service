from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request, Response

from src.common.auth.service_auth import verify_service_token
from src.common.schemas.common import HealthResponse, ModelInfoItem, ModelsResponse, ReadyResponse

router = APIRouter()

VERSION = "1.0.0"


def _model_items(model_manager) -> list[ModelInfoItem]:
    """Per-model identity (name + dims), shared by /ready and /v1/models.

    `type` is the role (embedder/reranker) so the admin dashboard can join it
    against the /ready `models` bool-map keys to show names + dims.
    """
    return [
        ModelInfoItem(
            name=model_manager.embedder.model_name,
            loaded=model_manager.embedder.is_loaded,
            type="embedder",
            dimensions=model_manager.embedder.dimensions
            if model_manager.embedder.is_loaded
            else None,
        ),
        ModelInfoItem(
            name=model_manager.reranker.model_name,
            loaded=model_manager.reranker.is_loaded,
            type="reranker",
            dimensions=None,
        ),
    ]


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

    # The reranker-role split deployment (TEMP — see ModelManager) only serves
    # /v1/rerank and never calls CMS, so its readiness must NOT gate on CMS —
    # otherwise it reports not_ready forever (cms unreachable/unset) and Cranl
    # treats the pod as unhealthy. Every other role keeps CMS in /ready.
    role = getattr(model_manager, "role", "api")
    requires_cms = role != "reranker"
    all_ready = model_manager.all_ready and (cms_reachable or not requires_cms)

    if not all_ready:
        response.status_code = 503
        response.headers["Retry-After"] = "10"

    return ReadyResponse(
        status="ok" if all_ready else "not_ready",
        models=models_status,
        dependencies={"cms": cms_reachable},
        models_detail=_model_items(model_manager),
    )


@router.get(
    "/v1/models",
    response_model=ModelsResponse,
    dependencies=[Depends(verify_service_token)],
)
async def models(request: Request) -> ModelsResponse:
    """Enrichment-Service model registry.

    Whisper + CLIP are reported by Media-Service's /v1/models endpoint.
    Enrichment hosts:
      - embedder: Qwen3-Embedding-0.6B (1024-dim dense text vectors, dense-only)
      - reranker: bge-reranker-v2-m3 (cross-encoder for /v1/feed/news/slide)
    """
    return ModelsResponse(models=_model_items(request.app.state.model_manager))
