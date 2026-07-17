"""News-feed slide cache.

Caches internal compatibility-helper `/v1/feed/news/slide` responses by a content-addressable key
(anchor + retrieval params). Backed by Redis, short TTL.

Why: the slide pipeline is the most expensive internal ranking helper — dense
kNN + a cross-encoder rerank inference. This helper is never the public
News-feed serving path; CMS owns live story assembly. A short TTL turns repeated
compatibility calls into a single Redis GET.

Mirrors `src/llm/clients/llm_cache.py`: stable SHA-256 key, get/set wrapped so a
flaky Redis only ever misses (never breaks the request), versioned prefix.
"""
from __future__ import annotations

import hashlib

from redis.asyncio import Redis

from src.common.utils.logging import get_logger

logger = get_logger(__name__)

# Bump to invalidate every cached slide (e.g. after a ranking-rule or rerank
# change that would make old slides wrong).
CACHE_VERSION = 2

KEY_PREFIX = "enrich:slide"


class SlideCache:
    def __init__(self, redis: Redis, default_ttl_sec: int = 300) -> None:
        self._redis = redis
        self._default_ttl = default_ttl_sec

    @staticmethod
    def make_key(
        anchor_content_id: str,
        k: int,
        types: list[str] | None,
        formats: list[str] | None,
        exclude_ids: list[str] | None,
        rerank: bool,
    ) -> str:
        """Stable cache key. Lists are sorted so call-order doesn't fragment it."""
        h = hashlib.sha256()
        payload = "\x1f".join(
            [
                str(CACHE_VERSION),
                anchor_content_id,
                str(k),
                ",".join(sorted(types or [])),
                ",".join(sorted(formats or [])),
                ",".join(sorted(exclude_ids or [])),
                "1" if rerank else "0",
            ]
        )
        h.update(payload.encode("utf-8"))
        return f"{KEY_PREFIX}:v{CACHE_VERSION}:{h.hexdigest()}"

    async def get(self, key: str) -> str | None:
        try:
            raw = await self._redis.get(key)
        except Exception as exc:
            # A flaky Redis must NEVER break the feed path. Log and miss.
            logger.warning("slide_cache_get_failed", error=str(exc))
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)

    async def set(self, key: str, value: str, ttl_sec: int | None = None) -> None:
        try:
            await self._redis.set(key, value, ex=ttl_sec or self._default_ttl)
        except Exception as exc:
            logger.warning("slide_cache_set_failed", error=str(exc))
