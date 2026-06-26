from fastapi import APIRouter, Depends, Request

from src.common.auth.service_auth import verify_service_token
from src.common.middleware.error_handler import ExtractionError
from src.common.utils.logging import get_logger
from src.common.utils.metrics import extractions_total
from src.extraction.schemas.extract import (
    ApplePodcastRelatedRequest,
    ApplePodcastRelatedResponse,
    ExtractRequest,
    ExtractResponse,
    FeedExtractResponse,
    TelegramChannelRequest,
    TelegramChannelResponse,
    TwitterProfileRequest,
    TwitterProfileResponse,
    TwitterRecommendationsRequest,
    TwitterRecommendationsResponse,
    YouTubeChannelRequest,
    YouTubeChannelResponse,
    YouTubeRelatedRequest,
    YouTubeRelatedResponse,
    YouTubeSearchRequest,
    YouTubeSearchResponse,
)
from src.extraction.services.apple_podcast import ApplePodcastService
from src.extraction.services.extraction import ExtractionService
from src.extraction.services.telegram_channel import TelegramChannelService
from src.extraction.services.twitter_profile import TwitterProfileService
from src.extraction.services.youtube_innertube import YouTubeInnerTubeService

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


@router.post("/extract/twitter", response_model=TwitterProfileResponse)
async def extract_twitter(
    body: TwitterProfileRequest, request: Request
) -> TwitterProfileResponse:
    """Scrape an X profile's public syndication timeline.

    Powers Aggregation's Source Intelligence interaction-graph (retweet/quote/
    mention edges + followers) and X ingestion/preview (full recent tweets).
    """
    settings = request.app.state.settings
    service = TwitterProfileService(timeout_sec=settings.EXTRACT_TIMEOUT_SEC)

    try:
        return await service.fetch(body.username)
    except ExtractionError:
        raise
    except Exception as exc:
        extractions_total.labels(status="failure").inc()
        logger.error("twitter_extraction_failed", username=body.username, error=str(exc))
        raise ExtractionError(f"Twitter extraction failed: {exc}") from exc


@router.post(
    "/extract/twitter/recommendations", response_model=TwitterRecommendationsResponse
)
async def extract_twitter_recommendations(
    body: TwitterRecommendationsRequest, request: Request
) -> TwitterRecommendationsResponse:
    """X "who to follow" / قد يعجبك for a seed account (guest-accessible REST).

    Powers the Source Intelligence relatedness signal: feeding a trusted source
    returns accounts X considers similar — candidate sources to score + promote.
    """
    settings = request.app.state.settings
    service = TwitterProfileService(timeout_sec=settings.EXTRACT_TIMEOUT_SEC)

    try:
        return await service.fetch_recommendations(body.seed, body.limit)
    except ExtractionError:
        raise
    except Exception as exc:
        extractions_total.labels(status="failure").inc()
        logger.error("twitter_recs_failed", seed=body.seed, error=str(exc))
        raise ExtractionError(f"Twitter recommendations failed: {exc}") from exc


@router.post("/extract/youtube", response_model=YouTubeChannelResponse)
async def extract_youtube(
    body: YouTubeChannelRequest, request: Request
) -> YouTubeChannelResponse:
    """Read a YouTube channel via guest InnerTube (no API key, no quota).

    Powers the media Source Intelligence graph: recent video titles (relevance)
    + subscriber count (authority) for a candidate/seed channel.
    """
    settings = request.app.state.settings
    service = YouTubeInnerTubeService(timeout_sec=settings.EXTRACT_TIMEOUT_SEC)

    try:
        return await service.fetch_channel(body.channel)
    except ExtractionError:
        raise
    except Exception as exc:
        extractions_total.labels(status="failure").inc()
        logger.error("youtube_extraction_failed", channel=body.channel, error=str(exc))
        raise ExtractionError(f"YouTube extraction failed: {exc}") from exc


@router.post("/extract/youtube/related", response_model=YouTubeRelatedResponse)
async def extract_youtube_related(
    body: YouTubeRelatedRequest, request: Request
) -> YouTubeRelatedResponse:
    """Channels in a seed channel's watch-next graph — the YouTube relatedness
    signal (the analog of forwards/retweets) feeding candidate discovery.
    """
    settings = request.app.state.settings
    service = YouTubeInnerTubeService(timeout_sec=settings.EXTRACT_TIMEOUT_SEC)

    try:
        return await service.fetch_related(body.channel)
    except ExtractionError:
        raise
    except Exception as exc:
        extractions_total.labels(status="failure").inc()
        logger.error("youtube_related_failed", channel=body.channel, error=str(exc))
        raise ExtractionError(f"YouTube related failed: {exc}") from exc


@router.post("/extract/youtube/search", response_model=YouTubeSearchResponse)
async def extract_youtube_search(
    body: YouTubeSearchRequest, request: Request
) -> YouTubeSearchResponse:
    """Topic/keyword channel search via guest InnerTube — the YouTube analog of
    the iTunes podcast keyword net, for interest-driven media discovery.
    """
    settings = request.app.state.settings
    service = YouTubeInnerTubeService(timeout_sec=settings.EXTRACT_TIMEOUT_SEC)

    try:
        return await service.search_channels(body.query, body.limit)
    except ExtractionError:
        raise
    except Exception as exc:
        extractions_total.labels(status="failure").inc()
        logger.error("youtube_search_failed", query=body.query, error=str(exc))
        raise ExtractionError(f"YouTube search failed: {exc}") from exc


@router.post("/extract/apple-podcast/related", response_model=ApplePodcastRelatedResponse)
async def extract_apple_related(
    body: ApplePodcastRelatedRequest, request: Request
) -> ApplePodcastRelatedResponse:
    """A podcast's Apple "You Might Also Like" / "Listeners Also Subscribed" shelf
    — the co-listen relation, scraped from the public show page (no token).
    """
    settings = request.app.state.settings
    service = ApplePodcastService(timeout_sec=settings.EXTRACT_TIMEOUT_SEC)

    try:
        return await service.fetch_related(body.collection_id, body.country)
    except ExtractionError:
        raise
    except Exception as exc:
        extractions_total.labels(status="failure").inc()
        logger.error("apple_related_failed", collection_id=body.collection_id, error=str(exc))
        raise ExtractionError(f"Apple related failed: {exc}") from exc
