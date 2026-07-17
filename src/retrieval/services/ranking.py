"""Ranking rules for News-feed slide assembly (Slice B).

Pure functions, no I/O, no model inference. Each rule takes the
reranker's output list and returns a filtered + reordered list. Designed
to be chained in order: freshness → diversity → quotas.

These rules turn "AI relevance ranking" into "product experience" — the
reranker says what's relevant, the rules say what's READABLE in a feed.
Without them you get 3 tweets from the same account about an event from
6 hours ago. With them you get a varied, fresh, on-topic slice.
"""
import math
from collections import defaultdict
from datetime import datetime, timezone

from src.common.config import Settings
from src.common.utils.logging import get_logger
from src.common.utils.metrics import ranking_rules_dropped_total
from src.retrieval.schemas.related import RelatedItem

logger = get_logger(__name__)

# NEWS formats (and non-NEWS kinds for generic retrieval) map to settings
# attributes. Centralizing the mapping keeps freshness semantics consistent.
_DECAY_ATTR_BY_TYPE: dict[str, str] = {
    "TWEET":   "FRESHNESS_DECAY_TWEET_DAYS",
    "COMMENT": "FRESHNESS_DECAY_COMMENT_DAYS",
    "ARTICLE": "FRESHNESS_DECAY_ARTICLE_DAYS",
    "PODCAST": "FRESHNESS_DECAY_PODCAST_DAYS",
    "VIDEO":   "FRESHNESS_DECAY_VIDEO_DAYS",
}


def apply_freshness_decay(
    items: list[RelatedItem], settings: Settings
) -> list[RelatedItem]:
    """Multiply each item's score by exp(-age_days / τ).

    τ varies by NEWS format or content kind (configured in Settings) — tweets decay in
    days, podcasts in months. Items missing `published_at` skip the decay
    (don't penalize unknown-age items; rely on relevance alone for them).

    Reorders the list by decayed score. Doesn't drop items.
    """
    now = datetime.now(timezone.utc)
    decayed: list[RelatedItem] = []
    skipped = 0
    for item in items:
        kind_or_format = (
            item.content_format
            if item.content_type == "NEWS" and item.content_format
            else item.content_type
        )
        tau_attr = _DECAY_ATTR_BY_TYPE.get(kind_or_format or "")
        if not tau_attr or not item.published_at:
            # No decay configured for this type, or no published_at — keep
            # original score. (Article-only feeds in early-stage CMS may
            # have many items without published_at.)
            decayed.append(item)
            skipped += 1
            continue

        try:
            published = _parse_iso(item.published_at)
        except ValueError:
            decayed.append(item)
            skipped += 1
            continue

        age_days = max(0.0, (now - published).total_seconds() / 86400.0)
        tau_days = getattr(settings, tau_attr)
        if tau_days <= 0:
            decayed.append(item)
            continue

        factor = math.exp(-age_days / tau_days)
        new_score = round(item.score * factor, 6)
        decayed.append(item.model_copy(update={"score": new_score}))

    if skipped:
        logger.debug("freshness_decay_skipped", count=skipped)

    decayed.sort(key=lambda r: (-r.score, r.content_id))
    return decayed


def apply_source_diversity(
    items: list[RelatedItem], max_per_source: int
) -> list[RelatedItem]:
    """Cap items from the same `source_name` in the final list.

    Walks the list in rank order; for each item, drops it if we've already
    accepted `max_per_source` items from the same source. Preserves rank
    for accepted items.

    Items missing `source_name` aren't capped (treated as "unknown source").
    """
    if max_per_source <= 0:
        return items

    counts: dict[str, int] = defaultdict(int)
    kept: list[RelatedItem] = []
    dropped = 0
    for item in items:
        src = item.source_name
        if src is None:
            kept.append(item)
            continue
        if counts[src] >= max_per_source:
            dropped += 1
            continue
        counts[src] += 1
        kept.append(item)

    if dropped:
        ranking_rules_dropped_total.labels(rule="source_diversity").inc(dropped)
    return kept


def apply_format_quotas(
    items: list[RelatedItem], required_formats: list[str] | None
) -> list[RelatedItem]:
    """Filter to required NEWS formats and interleave to surface a mix.

    Items whose content_format is not in `required_formats` are DROPPED.

    Reorders to interleave formats when possible (best-rank item per required
    format first, then second-best, ...).

    `required_formats=None` or empty is a no-op (caller didn't ask for a mix).
    """
    if not required_formats:
        return items

    by_format: dict[str, list[RelatedItem]] = defaultdict(list)
    dropped = 0
    for item in items:
        content_format = item.content_format
        if item.content_type == "NEWS" and content_format in required_formats:
            by_format[content_format].append(item)
        else:
            dropped += 1

    if dropped:
        ranking_rules_dropped_total.labels(rule="format_quotas").inc(dropped)

    # Round-robin interleave across required formats in caller order.
    interleaved: list[RelatedItem] = []
    while any(by_format[format_] for format_ in required_formats):
        for format_ in required_formats:
            if by_format[format_]:
                interleaved.append(by_format[format_].pop(0))

    return interleaved


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp into an aware datetime in UTC.

    Handles `Z` suffix, explicit `+00:00`, and naive inputs (no tz). Naive
    inputs are treated as UTC — otherwise `.astimezone()` would reinterpret
    them in the server's local time and silently corrupt age_days.

    Raises ValueError on malformed input — callers should fall through to
    "no decay" on failure.
    """
    # Swap only the single trailing 'Z' (rstrip would consume multiple).
    s = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
    parsed = datetime.fromisoformat(s)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
