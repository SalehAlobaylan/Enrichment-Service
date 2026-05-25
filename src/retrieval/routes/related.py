"""POST /v1/related — hybrid retrieval endpoint (Slice A).

Embeds the query (or fetches stored embeddings by content_id), fans out to
CMS for dense + sparse kNN in parallel, fuses via RRF, returns ranked items.
"""
from fastapi import APIRouter, Depends, HTTPException, Request

from src.common.auth.service_auth import verify_service_token
from src.common.middleware.error_handler import EmbeddingError
from src.common.utils.logging import get_logger
from src.common.utils.metrics import related_requests_total
from src.retrieval.schemas.related import RelatedRequest, RelatedResponse
from src.retrieval.services.related import RelatedService

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/related", response_model=RelatedResponse)
async def related(body: RelatedRequest, request: Request) -> RelatedResponse:
    model_manager = request.app.state.model_manager
    cms_client = request.app.state.cms_client
    settings = request.app.state.settings

    if not model_manager.embedder.is_loaded:
        raise EmbeddingError("Embedding model is not loaded")

    service = RelatedService(model_manager.embedder, cms_client, settings)

    try:
        return await service.related(body)
    except ValueError as exc:
        # 404 if the anchor content_id doesn't exist or has no dense embedding
        # — distinct from a 5xx so the caller can decide whether to retry or
        # treat it as terminal (typically terminal).
        related_requests_total.labels(status="failure").inc()
        logger.warning("related_anchor_invalid", error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        related_requests_total.labels(status="failure").inc()
        logger.error("related_failed", error=str(exc))
        raise EmbeddingError(f"Related lookup failed: {exc}") from exc
