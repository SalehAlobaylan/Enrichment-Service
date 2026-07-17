"""Apple Podcasts "Listeners Also Subscribed" reader (co-listen relation).

Apple's iTunes Search API has no "similar" endpoint, but the public show page
(`podcasts.apple.com/<cc>/podcast/id<adamId>`) server-renders a "You Might Also
Like" shelf inside a `<script id="serialized-server-data">` JSON blob — no token,
no login. Each related item carries the related show's `adamId`, which the
Aggregation side resolves to an RSS feed via the iTunes lookup API. This is the
podcast analog of the X recommendations / Telegram forward signals.

Fetched through Scrapling (curl_cffi browser-impersonation) — the same stealth
boundary as the Telegram/Twitter readers. Failures degrade to exists=False.
"""

import asyncio
import json
import re

from src.common.utils.logging import get_logger
from src.common.utils.url_guard import validate_public_url
from src.common.workload_admission import WorkloadExecutors
from src.extraction.schemas.extract import (
    ApplePodcastRelatedResponse,
    AppleRelatedShow,
)
from src.extraction.services.extraction import _get_fetcher

logger = get_logger(__name__)

_SERVER_DATA_RE = re.compile(
    r'<script[^>]*id="serialized-server-data"[^>]*>(.*?)</script>', re.S
)


def _walk(obj, cb, depth: int = 0):
    if depth > 32:
        return
    if isinstance(obj, dict):
        cb(obj)
        for v in obj.values():
            _walk(v, cb, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, cb, depth + 1)


_TITLE_HINTS = ("also", "might", "like", "يعجبك", "أيضا", "ايضا", "مشابه")


def _find_related_shelf(data) -> dict | None:
    """The "You Might Also Like" / "Listeners Also Subscribed" shelf. Primary
    match is the shelf title; falls back to a language-independent structural
    match (a shelf whose items are podcast-SHOW lockups — they carry the
    show-level `releaseFrequency` + `adamId`, unlike episode lockups) so a
    localized storefront whose title isn't English still resolves."""
    by_title: list[dict] = []
    by_shape: list[dict] = []

    def check(o: dict):
        items = o.get("items")
        if not (isinstance(items, list) and items):
            return
        t = o.get("title") or o.get("segueTitle")
        if isinstance(t, str) and any(k in t.lower() for k in _TITLE_HINTS):
            by_title.append(o)
            return
        first = items[0]
        if isinstance(first, dict) and first.get("adamId") and "releaseFrequency" in first:
            by_shape.append(o)

    _walk(data, check)
    if by_title:
        return by_title[0]
    return by_shape[0] if by_shape else None


def _show_title(item: dict) -> str | None:
    t = item.get("titleAccessibilityLabel")
    if isinstance(t, str) and t.strip():
        return t.strip()
    t = item.get("title")
    if isinstance(t, str) and t.strip():
        return t.strip()
    if isinstance(t, dict):
        c = t.get("content")
        if isinstance(c, str):
            return c.strip()
    return None


class ApplePodcastService:
    """Scrape a podcast's Apple "You Might Also Like" shelf → related adamIds."""

    def __init__(
        self, timeout_sec: int = 30, executors: WorkloadExecutors | None = None
    ) -> None:
        self.timeout_sec = timeout_sec
        self.executors = executors

    async def fetch_related(
        self, collection_id: str, country: str = "us"
    ) -> ApplePodcastRelatedResponse:
        if self.executors is not None:
            return await self.executors.run(
                "extraction", self._do_fetch_related, collection_id, country
            )
        return await asyncio.to_thread(self._do_fetch_related, collection_id, country)

    def _do_fetch_related(
        self, collection_id: str, country: str
    ) -> ApplePodcastRelatedResponse:
        cid = re.sub(r"\D", "", collection_id or "")
        cc = re.sub(r"[^a-z]", "", (country or "us").lower())[:2] or "us"
        if not cid:
            return ApplePodcastRelatedResponse(collection_id=collection_id, exists=False)

        url = f"https://podcasts.apple.com/{cc}/podcast/id{cid}"
        try:
            validate_public_url(url)
            page = _get_fetcher().get(url, timeout=self.timeout_sec, stealthy_headers=True)
        except Exception as exc:
            logger.debug("apple_fetch_failed", collection_id=cid, error=str(exc))
            return ApplePodcastRelatedResponse(collection_id=cid, exists=False)

        raw = page.body or b""
        html = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else str(raw)
        m = _SERVER_DATA_RE.search(html)
        if not m:
            return ApplePodcastRelatedResponse(collection_id=cid, exists=False)
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            return ApplePodcastRelatedResponse(collection_id=cid, exists=False)

        shelf = _find_related_shelf(data)
        related: list[AppleRelatedShow] = []
        seen: set[str] = set()
        for it in (shelf or {}).get("items", []):
            aid = it.get("adamId")
            if not aid:
                continue
            aid = str(aid)
            if aid == cid or aid in seen:
                continue
            seen.add(aid)
            related.append(
                AppleRelatedShow(
                    adam_id=aid,
                    title=_show_title(it),
                    genres=[g for g in (it.get("genreNames") or []) if isinstance(g, str)],
                )
            )
        return ApplePodcastRelatedResponse(
            collection_id=cid, exists=True, related=related[:25]
        )
