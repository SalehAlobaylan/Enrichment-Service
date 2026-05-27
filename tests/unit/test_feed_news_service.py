"""Unit tests for FeedNewsService — News-feed slide orchestration."""
from unittest.mock import AsyncMock

import pytest

from src.common.config import Settings
from src.retrieval.schemas.feed_news import FeedNewsSlideRequest
from src.retrieval.schemas.related import RelatedItem, RelatedResponse
from src.retrieval.services.feed_news import FeedNewsService


@pytest.fixture
def settings() -> Settings:
    return Settings(
        SERVICE_AUTH_TOKEN="t",
        CMS_SERVICE_TOKEN="c",
        CMS_BASE_URL="x",
        RERANK_INPUT_K=10,
        NEWS_MAX_PER_SOURCE=2,
    )


@pytest.fixture
def mock_related() -> AsyncMock:
    """Mock RelatedService that returns a deterministic batch of related items."""
    related = AsyncMock()
    return related


@pytest.fixture
def mock_cms() -> AsyncMock:
    cms = AsyncMock()
    cms.get_content_item_basic.return_value = {
        "id": "anchor-1",
        "type": "ARTICLE",
        "title": "Anchor article",
        "excerpt": "anchor excerpt",
        "source_name": "anchor-source",
        "published_at": "2026-05-20T12:00:00Z",
    }
    return cms


@pytest.fixture
def service(mock_related, mock_cms, settings) -> FeedNewsService:
    return FeedNewsService(mock_related, mock_cms, settings)


@pytest.mark.asyncio
async def test_slide_returns_anchor_plus_related(
    service: FeedNewsService, mock_related: AsyncMock
) -> None:
    mock_related.related.return_value = RelatedResponse(
        results=[
            RelatedItem(
                content_id=f"r{i}",
                score=1.0 - i * 0.1,
                content_type="TWEET",
                sources=["dense", "sparse"],
                source_name=f"src-{i}",
            )
            for i in range(5)
        ]
    )

    resp = await service.slide(
        FeedNewsSlideRequest(anchor_content_id="anchor-1", k=3)
    )

    assert resp.anchor.content_id == "anchor-1"
    assert resp.anchor.type == "ARTICLE"
    assert len(resp.related) == 3  # k truncation


@pytest.mark.asyncio
async def test_slide_excludes_anchor_from_related(
    service: FeedNewsService, mock_related: AsyncMock
) -> None:
    mock_related.related.return_value = RelatedResponse(results=[])

    await service.slide(
        FeedNewsSlideRequest(
            anchor_content_id="anchor-1",
            k=3,
            exclude_ids=["already-shown"],
        )
    )

    # The RelatedRequest passed to RelatedService should carry both the
    # auto-excluded anchor + the caller's exclude_ids.
    related_request = mock_related.related.call_args.args[0]
    assert related_request.content_id == "anchor-1"
    assert "anchor-1" in related_request.exclude_ids
    assert "already-shown" in related_request.exclude_ids


@pytest.mark.asyncio
async def test_slide_applies_source_diversity_before_truncate(
    service: FeedNewsService, mock_related: AsyncMock
) -> None:
    """With NEWS_MAX_PER_SOURCE=2 and 3 items from the same source, the 3rd
    is dropped — the remaining slide has only 2 (even though k=3)."""
    mock_related.related.return_value = RelatedResponse(
        results=[
            RelatedItem(
                content_id="a", score=0.9, content_type="TWEET",
                sources=["dense"], source_name="src-x",
            ),
            RelatedItem(
                content_id="b", score=0.8, content_type="TWEET",
                sources=["dense"], source_name="src-x",
            ),
            RelatedItem(
                content_id="c", score=0.7, content_type="TWEET",
                sources=["dense"], source_name="src-x",  # over cap
            ),
        ]
    )
    resp = await service.slide(
        FeedNewsSlideRequest(anchor_content_id="anchor-1", k=3)
    )
    # Diversity drops the 3rd src-x item; nothing left to fill the 3rd slot.
    assert [r.content_id for r in resp.related] == ["a", "b"]


@pytest.mark.asyncio
async def test_slide_default_types_to_tweet_and_comment(
    service: FeedNewsService, mock_related: AsyncMock
) -> None:
    mock_related.related.return_value = RelatedResponse(results=[])
    await service.slide(FeedNewsSlideRequest(anchor_content_id="anchor-1", k=3))
    related_request = mock_related.related.call_args.args[0]
    assert related_request.types == ["TWEET", "COMMENT"]


@pytest.mark.asyncio
async def test_slide_caller_types_override_default(
    service: FeedNewsService, mock_related: AsyncMock
) -> None:
    mock_related.related.return_value = RelatedResponse(results=[])
    await service.slide(
        FeedNewsSlideRequest(
            anchor_content_id="anchor-1", k=3, types=["COMMENT"]
        )
    )
    related_request = mock_related.related.call_args.args[0]
    assert related_request.types == ["COMMENT"]


@pytest.mark.asyncio
async def test_slide_requests_oversize_k_for_rule_headroom(
    service: FeedNewsService, mock_related: AsyncMock, settings: Settings
) -> None:
    """The slide endpoint asks for RERANK_INPUT_K candidates from
    RelatedService, NOT the caller's k. The rules need headroom to drop
    items without running short."""
    mock_related.related.return_value = RelatedResponse(results=[])
    await service.slide(FeedNewsSlideRequest(anchor_content_id="anchor-1", k=3))
    related_request = mock_related.related.call_args.args[0]
    assert related_request.k == settings.RERANK_INPUT_K  # 10 per fixture


@pytest.mark.asyncio
async def test_slide_rerank_always_on(
    service: FeedNewsService, mock_related: AsyncMock
) -> None:
    mock_related.related.return_value = RelatedResponse(results=[])
    await service.slide(FeedNewsSlideRequest(anchor_content_id="anchor-1", k=3))
    related_request = mock_related.related.call_args.args[0]
    assert related_request.rerank is True
