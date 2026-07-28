"""News-feed slide assembly orchestrator (Slice B).

Pipeline:
  1. Resolve anchor — fetch the full content item from CMS (for the
     response payload) AND from the embeddings endpoint (for retrieval).
  2. Dense retrieval + rerank via RelatedService — large k so the
     ranking rules have headroom to drop items without running short.
  3. Apply ranking rules in order:
       - freshness decay  (rescore by content-type-specific τ)
       - source diversity (cap per source_name)
       - format quotas    (interleave to ensure required NEWS-format mix)
  4. Truncate to caller's k.

This service does no model inference itself — it composes RelatedService
+ pure ranking functions. Slice C will add FeedPodsService following
the same pattern, with personalization rules instead of freshness/diversity.
"""
from src.common.clients.cms import CMSClient
from src.common.config import Settings
from src.common.utils.logging import get_logger
from src.common.utils.metrics import (
    feed_news_duration,
    feed_news_requests_total,
    feed_slide_cache_hits_total,
    feed_slide_cache_misses_total,
)
from src.retrieval.clients.slide_cache import SlideCache
from src.retrieval.schemas.feed_news import (
    FeedNewsAnchor,
    FeedNewsSlideRequest,
    FeedNewsSlideResponse,
)
from src.retrieval.schemas.related import RelatedRequest
from src.retrieval.services import ranking
from src.retrieval.services.related import RelatedService

logger = get_logger(__name__)

# The canonical News slide operates on NEWS kind with format-specific mix.
_DEFAULT_TYPES = ["NEWS"]
_DEFAULT_FORMATS = ["ARTICLE", "TWEET", "COMMENT"]


class FeedNewsService:
    def __init__(
        self,
        related_service: RelatedService,
        cms_client: CMSClient,
        settings: Settings,
        cache: SlideCache | None = None,
    ) -> None:
        self.related = related_service
        self.cms_client = cms_client
        self.settings = settings
        self.cache = cache

    async def slide(self, req: FeedNewsSlideRequest) -> FeedNewsSlideResponse:
        # Cache lookup — the assembled slide is keyed by anchor + retrieval
        # params. A hit skips the whole pipeline (2× kNN + rerank), which is
        # what keeps the cross-encoder off the synchronous feed path.
        cache_key: str | None = None
        if self.cache is not None:
            cache_key = SlideCache.make_key(
                req.anchor_content_id,
                req.k,
                req.types,
                req.formats,
                req.exclude_ids,
                rerank=True,
            )
            cached = await self.cache.get(cache_key)
            if cached is not None:
                try:
                    resp = FeedNewsSlideResponse.model_validate_json(cached)
                    feed_slide_cache_hits_total.inc()
                    feed_news_requests_total.labels(status="success").inc()
                    logger.info("feed_news_slide_cache_hit", anchor=req.anchor_content_id)
                    return resp
                except Exception as exc:
                    logger.warning("feed_news_slide_cache_decode_failed", error=str(exc))
            feed_slide_cache_misses_total.inc()

        with feed_news_duration.time():
            # 1. Anchor — fetch the basic record from CMS for the response payload.
            anchor_payload = await self.cms_client.get_content_item_basic(
                req.anchor_content_id
            )
            anchor = self._anchor_from_payload(anchor_payload)

            # 2. Dense retrieval + rerank. We over-fetch (RERANK_INPUT_K)
            # because the ranking rules below WILL drop items and we want
            # the final list to still have `k` worth of survivors.
            related_resp = await self.related.related(
                RelatedRequest(
                    content_id=req.anchor_content_id,
                    types=req.types or _DEFAULT_TYPES,
                    formats=req.formats or _DEFAULT_FORMATS,
                    # Auto-exclude the anchor itself, then any caller-supplied ids.
                    exclude_ids=[req.anchor_content_id, *(req.exclude_ids or [])],
                    k=self.settings.RERANK_INPUT_K,
                    rerank=True,
                )
            )

            # 3. Ranking rules (freshness first — it rescores; diversity
            # then format quotas filter/reorder the rescored list).
            items = ranking.apply_freshness_decay(related_resp.results, self.settings)
            items = ranking.apply_source_diversity(
                items, self.settings.NEWS_MAX_PER_SOURCE
            )
            items = ranking.apply_format_quotas(items, req.formats or _DEFAULT_FORMATS)

            # 4. Truncate to the caller's k.
            final = items[: req.k]

        resp = FeedNewsSlideResponse(anchor=anchor, related=final)

        # Store for subsequent identical feed loads (short TTL).
        if self.cache is not None and cache_key is not None:
            await self.cache.set(cache_key, resp.model_dump_json())

        feed_news_requests_total.labels(status="success").inc()
        logger.info(
            "feed_news_slide_complete",
            anchor=req.anchor_content_id,
            requested=req.k,
            after_rerank=len(related_resp.results),
            after_rules=len(items),
            returned=len(final),
        )
        return resp

    @staticmethod
    def _anchor_from_payload(payload: dict) -> FeedNewsAnchor:
        """Map CMS's full content-item payload to the lightweight anchor.

        Tolerant to missing fields — CMS may evolve which keys are
        populated for a given item; we surface what we have and leave the
        rest None. Required: id + type (otherwise the slide is meaningless).
        """
        return FeedNewsAnchor(
            content_id=str(payload.get("id") or payload.get("ID")),
            type=str(payload.get("type") or ""),
            title=payload.get("title"),
            excerpt=payload.get("excerpt"),
            source_name=payload.get("source_name"),
            published_at=payload.get("published_at"),
        )
