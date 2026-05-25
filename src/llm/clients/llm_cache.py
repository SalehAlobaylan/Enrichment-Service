"""LLM response cache.

Caches completed LLM calls by content-addressable key (SHA-256 of the prompt
+ model + sampling params). Backed by Redis, default 7-day TTL.

Used by LLMClient.complete() to skip the upstream provider when an identical
prompt has been seen before. Translation and summarization are pure functions
of their inputs, so this is a safe and high-leverage optimization.
"""
from __future__ import annotations

import hashlib

from redis.asyncio import Redis

from src.common.utils.logging import get_logger

logger = get_logger(__name__)

# Bump to invalidate every cached response (e.g., after a prompt-template
# change that would make old responses semantically wrong).
CACHE_VERSION = 1

# Keys are prefixed so the same Redis instance can serve other consumers
# (arq, future caches) without collision.
KEY_PREFIX = "enrich:llm"


class LLMCache:
    def __init__(self, redis: Redis, default_ttl_sec: int = 7 * 86400) -> None:
        self._redis = redis
        self._default_ttl = default_ttl_sec

    @staticmethod
    def make_key(
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Stable cache key for a deterministic LLM call."""
        h = hashlib.sha256()
        # Field order is fixed; field separator chosen so injected ":" in a
        # prompt can't collide with another field's hash slot.
        payload = "\x1f".join(
            [
                str(CACHE_VERSION),
                provider,
                model,
                system_prompt,
                user_prompt,
                str(max_tokens),
                f"{temperature:.4f}",
            ]
        )
        h.update(payload.encode("utf-8"))
        return f"{KEY_PREFIX}:v{CACHE_VERSION}:{h.hexdigest()}"

    async def get(self, key: str) -> str | None:
        try:
            raw = await self._redis.get(key)
        except Exception as exc:
            # A flaky Redis must NEVER break the LLM path. Log and miss.
            logger.warning("llm_cache_get_failed", error=str(exc))
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
            logger.warning("llm_cache_set_failed", error=str(exc))
