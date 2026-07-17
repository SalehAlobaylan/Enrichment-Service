"""LLM response cache.

Caches completed LLM calls by content-addressable key (SHA-256 of the prompt
+ model + sampling params). Backed by Redis, default 7-day TTL.

Used by LLMClient.complete() to skip the upstream provider when an identical
prompt has been seen before. Translation and summarization are pure functions
of their inputs, so this is a safe and high-leverage optimization.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

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
        entry = await self.get_entry(key)
        return entry[0] if entry else None

    async def get_entry(self, key: str) -> tuple[str, dict[str, Any] | None] | None:
        """Return response + optional metering provenance.

        Legacy values are plain strings and intentionally remain cache hits;
        callers can label their avoided-cost estimate accordingly.
        """
        try:
            raw = await self._redis.get(key)
        except Exception as exc:
            # A flaky Redis must NEVER break the LLM path. Log and miss.
            logger.warning("llm_cache_get_failed", error=str(exc))
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        text = str(raw)
        try:
            decoded = json.loads(text)
            if (
                isinstance(decoded, dict)
                and decoded.get("version") == 2
                and isinstance(decoded.get("text"), str)
            ):
                usage = decoded.get("usage")
                return decoded["text"], usage if isinstance(usage, dict) else None
        except (TypeError, ValueError):
            pass
        return text, None

    async def set(self, key: str, value: str, ttl_sec: int | None = None) -> None:
        try:
            await self._redis.set(key, value, ex=ttl_sec or self._default_ttl)
        except Exception as exc:
            logger.warning("llm_cache_set_failed", error=str(exc))

    async def set_entry(
        self,
        key: str,
        value: str,
        usage: dict[str, int],
        ttl_sec: int | None = None,
    ) -> None:
        """Store a versioned entry with the provider response's measured units."""
        try:
            payload = json.dumps(
                {"version": 2, "text": value, "usage": usage}, separators=(",", ":")
            )
            await self._redis.set(key, payload, ex=ttl_sec or self._default_ttl)
        except Exception as exc:
            logger.warning("llm_cache_set_failed", error=str(exc))
