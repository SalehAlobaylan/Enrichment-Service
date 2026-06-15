from fastapi import APIRouter, Depends, Request

from src.common.auth.service_auth import verify_service_token
from src.common.middleware.error_handler import ExtractionError
from src.common.utils.logging import get_logger
from src.common.utils.metrics import extractions_total
from src.extraction.schemas.extract import (
    ExtractRequest,
    ExtractResponse,
    FeedExtractResponse,
    TelegramChannelRequest,
    TelegramChannelResponse,
)
from src.extraction.services.extraction import ExtractionService
from src.extraction.services.telegram_channel import TelegramChannelService

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/extract", response_model=ExtractResponse)
async def extract(body: ExtractRequest, request: Request) -> ExtractResponse:
    settings = request.app.state.settings
    service = ExtractionService(timeout_sec=settings.EXTRACT_TIMEOUT_SEC)

    try:
        return await service.extract(body.url, include_html=body.include_html)
    except ExtractionError:
        raise
    except Exception as exc:
        extractions_total.labels(status="failure").inc()
        logger.error("extraction_failed", url=body.url, error=str(exc))
        raise ExtractionError(f"Extraction failed: {exc}") from exc


@router.post("/extract/feed", response_model=FeedExtractResponse)
async def extract_feed(body: ExtractRequest, request: Request) -> FeedExtractResponse:
    """Extract EVERY item from an RSS/Atom feed (for whole-feed import)."""
    settings = request.app.state.settings
    service = ExtractionService(timeout_sec=settings.EXTRACT_TIMEOUT_SEC)

    try:
        return await service.extract_feed(body.url)
    except ExtractionError:
        raise
    except Exception as exc:
        extractions_total.labels(status="failure").inc()
        logger.error("feed_extraction_failed", url=body.url, error=str(exc))
        raise ExtractionError(f"Feed extraction failed: {exc}") from exc


@router.post("/extract/telegram", response_model=TelegramChannelResponse)
async def extract_telegram(
    body: TelegramChannelRequest, request: Request
) -> TelegramChannelResponse:
    """Scrape a Telegram channel's public preview (t.me/s/<username>).

    Powers Aggregation's Source Intelligence forward-graph for Telegram —
    returns recent posts + forwarded/mentioned channels + subscriber count.
    """
    settings = request.app.state.settings
    service = TelegramChannelService(timeout_sec=settings.EXTRACT_TIMEOUT_SEC)

    try:
        return await service.fetch(body.username)
    except ExtractionError:
        raise
    except Exception as exc:
        extractions_total.labels(status="failure").inc()
        logger.error("telegram_extraction_failed", username=body.username, error=str(exc))
        raise ExtractionError(f"Telegram extraction failed: {exc}") from exc
