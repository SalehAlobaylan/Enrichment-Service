"""Unit tests for ranking rules — freshness decay, source diversity, type quotas."""
from datetime import datetime, timedelta, timezone

import pytest

from src.common.config import Settings
from src.retrieval.schemas.related import RelatedItem
from src.retrieval.services import ranking


def _item(
    cid: str,
    score: float,
    content_type: str = "NEWS",
    content_format: str | None = "TWEET",
    source: str | None = None,
    age_days: float | None = None,
) -> RelatedItem:
    published_at = None
    if age_days is not None:
        ts = datetime.now(timezone.utc) - timedelta(days=age_days)
        published_at = ts.isoformat().replace("+00:00", "Z")
    return RelatedItem(
        content_id=cid,
        score=score,
        content_type=content_type,
        content_format=content_format,
        sources=["dense"],
        published_at=published_at,
        source_name=source,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        SERVICE_AUTH_TOKEN="t",
        CMS_SERVICE_TOKEN="c",
        CMS_BASE_URL="x",
        FRESHNESS_DECAY_TWEET_DAYS=2,
        FRESHNESS_DECAY_COMMENT_DAYS=2,
        FRESHNESS_DECAY_ARTICLE_DAYS=30,
        FRESHNESS_DECAY_PODCAST_DAYS=90,
        FRESHNESS_DECAY_VIDEO_DAYS=30,
        NEWS_MAX_PER_SOURCE=2,
    )


# ─── Freshness decay ──────────────────────────────────────


def test_freshness_decay_keeps_unaged_items_unchanged(settings) -> None:
    items = [_item("a", 1.0, "NEWS", "TWEET")]  # no published_at
    decayed = ranking.apply_freshness_decay(items, settings)
    assert decayed[0].score == 1.0


def test_freshness_decay_decays_old_items(settings) -> None:
    # TWEET τ=2 days → 4 days old = exp(-2) ≈ 0.135
    items = [_item("old", 1.0, "NEWS", "TWEET", age_days=4)]
    decayed = ranking.apply_freshness_decay(items, settings)
    assert decayed[0].score == pytest.approx(0.135335, abs=1e-4)


def test_freshness_decay_reorders_by_decayed_score(settings) -> None:
    # Tweet from 4 days ago (~0.135 after decay) vs. fresh score 0.5 →
    # the older but originally higher score loses to the fresh lower one.
    items = [
        _item("old_high", 1.0, "NEWS", "TWEET", age_days=4),
        _item("fresh_low", 0.5, "NEWS", "TWEET", age_days=0),
    ]
    decayed = ranking.apply_freshness_decay(items, settings)
    assert [r.content_id for r in decayed] == ["fresh_low", "old_high"]


def test_freshness_decay_uses_type_specific_tau(settings) -> None:
    # 30 days old ARTICLE: τ=30 → exp(-1) ≈ 0.368
    # 30 days old TWEET:   τ=2  → exp(-15) ≈ 3.06e-7
    article = _item("article", 1.0, "NEWS", "ARTICLE", age_days=30)
    tweet = _item("tweet", 1.0, "NEWS", "TWEET", age_days=30)
    decayed = ranking.apply_freshness_decay([article, tweet], settings)
    article_decayed = next(r for r in decayed if r.content_id == "article")
    tweet_decayed = next(r for r in decayed if r.content_id == "tweet")
    assert article_decayed.score > tweet_decayed.score
    assert article_decayed.score == pytest.approx(0.367879, abs=1e-4)


# ─── Source diversity ──────────────────────────────────────


def test_source_diversity_caps_per_source() -> None:
    items = [
        _item(f"item_{i}", 1.0 - i * 0.01, "NEWS", "TWEET", source="acme")
        for i in range(5)
    ]
    kept = ranking.apply_source_diversity(items, max_per_source=2)
    assert len(kept) == 2
    assert kept[0].content_id == "item_0"  # best-scored survives
    assert kept[1].content_id == "item_1"


def test_source_diversity_preserves_rank() -> None:
    items = [
        _item("a", 0.9, "NEWS", "TWEET", source="src_x"),
        _item("b", 0.8, "NEWS", "TWEET", source="src_y"),
        _item("c", 0.7, "NEWS", "TWEET", source="src_x"),
        _item("d", 0.6, "NEWS", "TWEET", source="src_x"),  # 3rd from src_x → dropped
    ]
    kept = ranking.apply_source_diversity(items, max_per_source=2)
    assert [r.content_id for r in kept] == ["a", "b", "c"]


def test_source_diversity_doesnt_cap_unknown_source() -> None:
    items = [
        _item("a", 0.9, "NEWS", "TWEET", source=None),
        _item("b", 0.8, "NEWS", "TWEET", source=None),
        _item("c", 0.7, "NEWS", "TWEET", source=None),
    ]
    kept = ranking.apply_source_diversity(items, max_per_source=1)
    # source_name=None items are not capped (treated as unknown).
    assert len(kept) == 3


def test_source_diversity_max_zero_returns_all() -> None:
    items = [_item("a", 0.9, "NEWS", "TWEET", source="acme")]
    assert ranking.apply_source_diversity(items, max_per_source=0) == items


# ─── Type quotas ──────────────────────────────────────────


def test_format_quotas_interleave_required_formats() -> None:
    items = [
        _item("t1", 0.9, "NEWS", "TWEET"),
        _item("t2", 0.8, "NEWS", "TWEET"),
        _item("c1", 0.7, "NEWS", "COMMENT"),
        _item("t3", 0.6, "NEWS", "TWEET"),
        _item("c2", 0.5, "NEWS", "COMMENT"),
    ]
    # TWEET then COMMENT interleaved: t1, c1, t2, c2, t3
    result = ranking.apply_format_quotas(items, required_formats=["TWEET", "COMMENT"])
    assert [r.content_id for r in result] == ["t1", "c1", "t2", "c2", "t3"]


def test_format_quotas_no_op_when_none() -> None:
    items = [_item("a", 0.9), _item("b", 0.8, "NEWS", "COMMENT")]
    assert ranking.apply_format_quotas(items, required_formats=None) == items


def test_format_quotas_drops_off_format_items() -> None:
    items = [
        _item("t1", 0.9, "NEWS", "TWEET"),
        _item("article", 0.8, "NEWS", "ARTICLE"),
        _item("c1", 0.7, "NEWS", "COMMENT"),
    ]
    result = ranking.apply_format_quotas(items, required_formats=["TWEET", "COMMENT"])
    # Off-type items are dropped — they can't leak into the slide.
    assert [r.content_id for r in result] == ["t1", "c1"]
