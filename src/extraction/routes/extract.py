from collections.abc import Awaitable
from typing import TypeVar

from fastapi import APIRouter, Depends, Request

from src.common.auth.service_auth import verify_service_token
from src.common.middleware.error_handler import ExtractionError
from src.common.utils.logging import get_logger
from src.common.utils.metrics import extractions_total
from src.common.utils.url_guard import safe_url_label
from src.common.workload_admission import WorkloadOverloadedError
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
    YouTubeParseFeedRequest,
    YouTubeParseFeedResponse,
    YouTubePodcastSearchRequest,
    YouTubePodcastSearchResponse,
    YouTubeRelatedRequest,
    YouTubeRelatedResponse,
    YouTubeResolveLinksRequest,
    YouTubeResolveLinksResponse,
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
T = TypeVar("T")


async def _run_admitted(request: Request, operation: Awaitable[T]) -> T:
    async with request.app.state.workload_admission.acquire("extraction"):
        return await operation


def _executors(request: Request):
    return getattr(request.app.state, "workload_executors", None)


@router.post("/extract", response_model=ExtractResponse)
async def extract(body: ExtractRequest, request: Request) -> ExtractResponse:
    settings = request.app.state.settings
    service = ExtractionService(
        timeout_sec=settings.EXTRACT_TIMEOUT_SEC, executors=_executors(request)
    )

    try:
        return await _run_admitted(
            request, service.extract(body.url, include_html=body.include_html)
        )
    except (ExtractionError, WorkloadOverloadedError):
        raise
    except Exception:
        extractions_total.labels(status="failure").inc()
        logger.error("extraction_failed", url=safe_url_label(body.url))
        raise ExtractionError("Extraction failed")


@router.post("/extract/feed", response_model=FeedExtractResponse)
async def extract_feed(body: ExtractRequest, request: Request) -> FeedExtractResponse:
    """Extract EVERY item from an RSS/Atom feed (for whole-feed import)."""
    settings = request.app.state.settings
    service = ExtractionService(
        timeout_sec=settings.EXTRACT_TIMEOUT_SEC, executors=_executors(request)
    )

    try:
        return await _run_admitted(request, service.extract_feed(body.url))
    except (ExtractionError, WorkloadOverloadedError):
        raise
    except Exception:
        extractions_total.labels(status="failure").inc()
        logger.error("feed_extraction_failed", url=safe_url_label(body.url))
        raise ExtractionError("Feed extraction failed")


@router.post("/extract/telegram", response_model=TelegramChannelResponse)
async def extract_telegram(
    body: TelegramChannelRequest, request: Request
) -> TelegramChannelResponse:
    """Scrape a Telegram channel's public preview (t.me/s/<username>).

    Powers Aggregation's Source Intelligence forward-graph for Telegram —
    returns recent posts + forwarded/mentioned channels + subscriber count.
    """
    settings = request.app.state.settings
    service = TelegramChannelService(
        timeout_sec=settings.EXTRACT_TIMEOUT_SEC, executors=_executors(request)
    )

    try:
        return await _run_admitted(request, service.fetch(body.username))
    except (ExtractionError, WorkloadOverloadedError):
        raise
    except Exception:
        extractions_total.labels(status="failure").inc()
        logger.error("telegram_extraction_failed", username=body.username)
        raise ExtractionError("Telegram extraction failed")


@router.post("/extract/twitter", response_model=TwitterProfileResponse)
async def extract_twitter(
    body: TwitterProfileRequest, request: Request
) -> TwitterProfileResponse:
    """Scrape an X profile's public syndication timeline.

    Powers Aggregation's Source Intelligence interaction-graph (retweet/quote/
    mention edges + followers) and X ingestion/preview (full recent tweets).
    """
    settings = request.app.state.settings
    service = TwitterProfileService(
        timeout_sec=settings.EXTRACT_TIMEOUT_SEC, executors=_executors(request)
    )

    try:
        return await _run_admitted(request, service.fetch(body.username))
    except (ExtractionError, WorkloadOverloadedError):
        raise
    except Exception:
        extractions_total.labels(status="failure").inc()
        logger.error("twitter_extraction_failed", username=body.username)
        raise ExtractionError("Twitter extraction failed")


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
    service = TwitterProfileService(
        timeout_sec=settings.EXTRACT_TIMEOUT_SEC, executors=_executors(request)
    )

    try:
        return await _run_admitted(
            request, service.fetch_recommendations(body.seed, body.limit)
        )
    except (ExtractionError, WorkloadOverloadedError):
        raise
    except Exception:
        extractions_total.labels(status="failure").inc()
        logger.error("twitter_recs_failed", seed=body.seed)
        raise ExtractionError("Twitter recommendations failed")


@router.post("/extract/youtube", response_model=YouTubeChannelResponse)
async def extract_youtube(
    body: YouTubeChannelRequest, request: Request
) -> YouTubeChannelResponse:
    """Read a YouTube channel via guest InnerTube (no API key, no quota).

    Powers the media Source Intelligence graph: recent video titles (relevance)
    + subscriber count (authority) for a candidate/seed channel.
    """
    settings = request.app.state.settings
    service = YouTubeInnerTubeService(
        timeout_sec=settings.EXTRACT_TIMEOUT_SEC, executors=_executors(request)
    )

    try:
        return await _run_admitted(request, service.fetch_channel(body.channel))
    except (ExtractionError, WorkloadOverloadedError):
        raise
    except Exception:
        extractions_total.labels(status="failure").inc()
        logger.error("youtube_extraction_failed", channel=body.channel)
        raise ExtractionError("YouTube extraction failed")


@router.post("/extract/youtube/related", response_model=YouTubeRelatedResponse)
async def extract_youtube_related(
    body: YouTubeRelatedRequest, request: Request
) -> YouTubeRelatedResponse:
    """Channels in a seed channel's watch-next graph — the YouTube relatedness
    signal (the analog of forwards/retweets) feeding candidate discovery.
    """
    settings = request.app.state.settings
    service = YouTubeInnerTubeService(
        timeout_sec=settings.EXTRACT_TIMEOUT_SEC, executors=_executors(request)
    )

    try:
        return await _run_admitted(request, service.fetch_related(body.channel))
    except (ExtractionError, WorkloadOverloadedError):
        raise
    except Exception:
        extractions_total.labels(status="failure").inc()
        logger.error("youtube_related_failed", channel=body.channel)
        raise ExtractionError("YouTube related failed")


@router.post("/extract/youtube/search", response_model=YouTubeSearchResponse)
async def extract_youtube_search(
    body: YouTubeSearchRequest, request: Request
) -> YouTubeSearchResponse:
    """Topic/keyword channel search via guest InnerTube — the YouTube analog of
    the iTunes podcast keyword net, for interest-driven media discovery.
    """
    settings = request.app.state.settings
    service = YouTubeInnerTubeService(
        timeout_sec=settings.EXTRACT_TIMEOUT_SEC, executors=_executors(request)
    )

    try:
        return await _run_admitted(request, service.search_channels(body.query, body.limit))
    except (ExtractionError, WorkloadOverloadedError):
        raise
    except Exception:
        extractions_total.labels(status="failure").inc()
        logger.error("youtube_search_failed", query=body.query)
        raise ExtractionError("YouTube search failed")


@router.post(
    "/extract/youtube/podcast-search", response_model=YouTubePodcastSearchResponse
)
async def extract_youtube_podcast_search(
    body: YouTubePodcastSearchRequest, request: Request
) -> YouTubePodcastSearchResponse:
    """Podcast-intent guest search — parses YouTube's podcast shelves (not just
    channelRenderer) so famous podcast networks surface for media discovery.
    """
    settings = request.app.state.settings
    service = YouTubeInnerTubeService(
        timeout_sec=settings.EXTRACT_TIMEOUT_SEC, executors=_executors(request)
    )
    try:
        return await _run_admitted(request, service.search_podcasts(body.query, body.limit))
    except (ExtractionError, WorkloadOverloadedError):
        raise
    except Exception:
        extractions_total.labels(status="failure").inc()
        logger.error("youtube_podcast_search_failed", query=body.query)
        raise ExtractionError("YouTube podcast search failed")


@router.post("/extract/youtube/parse-feed", response_model=YouTubeParseFeedResponse)
async def extract_youtube_parse_feed(
    body: YouTubeParseFeedRequest, request: Request
) -> YouTubeParseFeedResponse:
    """Parse a pasted youtubei/v1 response into the channels it references — the
    manual seed path (the admin stays the authenticated session; we store no token).
    """
    settings = request.app.state.settings
    service = YouTubeInnerTubeService(
        timeout_sec=settings.EXTRACT_TIMEOUT_SEC, executors=_executors(request)
    )
    try:
        channels = await _run_admitted(request, service.parse_feed(body.raw))
        return YouTubeParseFeedResponse(channels=channels)
    except (ExtractionError, WorkloadOverloadedError):
        raise
    except Exception:
        extractions_total.labels(status="failure").inc()
        logger.error("youtube_parse_feed_failed")
        raise ExtractionError("YouTube parse feed failed")


@router.post(
    "/extract/youtube/resolve-links", response_model=YouTubeResolveLinksResponse
)
async def extract_youtube_resolve_links(
    body: YouTubeResolveLinksRequest, request: Request
) -> YouTubeResolveLinksResponse:
    """Resolve pasted YouTube links / @handles / video URLs into the channels behind
    them — the low-friction seed path (guest InnerTube, no auth, no quota)."""
    settings = request.app.state.settings
    service = YouTubeInnerTubeService(
        timeout_sec=settings.EXTRACT_TIMEOUT_SEC, executors=_executors(request)
    )
    try:
        channels = await _run_admitted(request, service.resolve_links(body.inputs))
        return YouTubeResolveLinksResponse(channels=channels)
    except (ExtractionError, WorkloadOverloadedError):
        raise
    except Exception:
        extractions_total.labels(status="failure").inc()
        logger.error("youtube_resolve_links_failed")
        raise ExtractionError("YouTube resolve links failed")


@router.post("/extract/apple-podcast/related", response_model=ApplePodcastRelatedResponse)
async def extract_apple_related(
    body: ApplePodcastRelatedRequest, request: Request
) -> ApplePodcastRelatedResponse:
    """A podcast's Apple "You Might Also Like" / "Listeners Also Subscribed" shelf
    — the co-listen relation, scraped from the public show page (no token).
    """
    settings = request.app.state.settings
    service = ApplePodcastService(
        timeout_sec=settings.EXTRACT_TIMEOUT_SEC, executors=_executors(request)
    )

    try:
        return await _run_admitted(
            request, service.fetch_related(body.collection_id, body.country)
        )
    except (ExtractionError, WorkloadOverloadedError):
        raise
    except Exception:
        extractions_total.labels(status="failure").inc()
        logger.error("apple_related_failed", collection_id=body.collection_id)
        raise ExtractionError("Apple related failed")
