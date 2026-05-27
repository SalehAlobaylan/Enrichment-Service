"""POST /v1/feed/news/slide — News-feed slide assembly (Slice B).

One anchor article + up to k related TWEET/COMMENT items, fully ranked
and filtered for feed display. Composed of hybrid retrieval, cross-encoder
rerank, and editorial ranking rules (freshness, diversity, quotas).
"""
import httpx
from fastapi import APIRouter, Depends, HTTPException, Request

from src.common.auth.service_auth import verify_service_token
from src.common.middleware.error_handler import EmbeddingError
from src.common.utils.logging import get_logger
from src.common.utils.metrics import feed_news_requests_total
from src.retrieval.schemas.feed_news import (
    FeedNewsSlideRequest,
    FeedNewsSlideResponse,
)
from src.retrieval.services.feed_news import FeedNewsService
from src.retrieval.services.related import RelatedService

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/feed/news/slide", response_model=FeedNewsSlideResponse)
async def feed_news_slide(
    body: FeedNewsSlideRequest, request: Request
) -> FeedNewsSlideResponse:
    model_manager = request.app.state.model_manager
    cms_client = request.app.state.cms_client
    settings = request.app.state.settings

    if not model_manager.embedder.is_loaded:
        raise EmbeddingError("Embedding model is not loaded")

    related_service = RelatedService(
        model_manager.embedder,
        cms_client,
        settings,
        reranker=model_manager.reranker,
    )
    feed_service = FeedNewsService(related_service, cms_client, settings)

    try:
        return await feed_service.slide(body)
    except ValueError as exc:
        # Bad anchor — content_id not found or missing embeddings.
        feed_news_requests_total.labels(status="failure").inc()
        logger.warning("feed_news_anchor_invalid", error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        # CMS rejected the anchor lookup (typically 404). Translate that
        # status to our caller so it can decide whether to retry or give up.
        feed_news_requests_total.labels(status="failure").inc()
        status = exc.response.status_code if exc.response is not None else 502
        logger.warning(
            "feed_news_cms_status_error",
            cms_status=status,
            anchor=body.anchor_content_id,
        )
        raise HTTPException(
            status_code=404 if status == 404 else 502,
            detail=f"CMS lookup failed for anchor {body.anchor_content_id}",
        ) from exc
    except Exception as exc:
        feed_news_requests_total.labels(status="failure").inc()
        logger.error("feed_news_slide_failed", error=str(exc))
        raise EmbeddingError(f"Feed news slide failed: {exc}") from exc
