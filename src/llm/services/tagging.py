"""Topic tag + named-entity extraction via LLM.

Aggregation passes `extract_tags: true` on /v1/embed for long-form content
(ARTICLE/VIDEO/PODCAST). The extracted tags flow through to CMS's
content_items.topic_tags column via the existing store_embedding write-back.

Mirror of services/translation.py and services/summarization.py: short
system prompt, JSON-only output, _strip_fences safety on the parse.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.common.utils.logging import get_logger
from src.common.utils.metrics import tag_extractions_total
from src.llm.clients.llm import LLMClient

logger = get_logger(__name__)


def _strip_fences(text: str) -> str:
    """Strip ``` / ```json fences some LLMs wrap JSON responses in."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    return text.strip()


SYSTEM_PROMPT = """You are a content tagger for a social news platform.
Given the article/transcript text, extract:
- 3 to 5 short lowercase topic tags (single words or short hyphenated phrases, no #)
- Named entities, grouped by type: people, organizations, locations

Return ONLY a JSON object with this exact shape:
{
  "tags": ["tag1", "tag2", ...],
  "entities": {
    "people": ["..."],
    "organizations": ["..."],
    "locations": ["..."]
  }
}

If a category has no values, return an empty array. Do not include any other text."""


@dataclass
class TagsResult:
    tags: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)


class TaggingService:
    """Extract topic tags + entities from arbitrary text using the LLM client.

    The LLM client handles caching (#3) and provider fallback (#8), so callers
    get those benefits for free.
    """

    # Below this length, tagging is rarely useful and not worth the LLM cost.
    MIN_TEXT_LENGTH = 200

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def is_eligible(self, text: str) -> bool:
        """True if text is long enough to warrant a tag-extraction LLM call."""
        return len(text) >= self.MIN_TEXT_LENGTH

    async def extract(self, text: str) -> TagsResult:
        """Run extraction. Returns empty result on parse failure (best-effort)."""
        if not self.is_eligible(text):
            tag_extractions_total.labels(status="skipped_short").inc()
            return TagsResult()

        # Cap input length — full novel-length transcripts would cost a fortune
        # without improving tag quality. 4000 chars is enough context.
        user_prompt = f"Tag the following text:\n\n{text[:4000]}"

        try:
            raw = await self.llm.complete(
                SYSTEM_PROMPT,
                user_prompt,
                max_tokens=512,
                temperature=0.1,
                operation="tag_extract",
            )
        except Exception as exc:
            logger.warning("tag_extraction_llm_failed", error=str(exc))
            tag_extractions_total.labels(status="llm_failed").inc()
            return TagsResult()

        try:
            parsed = json.loads(_strip_fences(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("tag_extraction_parse_failed", error=str(exc), raw=raw[:200])
            tag_extractions_total.labels(status="parse_failed").inc()
            return TagsResult()

        tags = parsed.get("tags") or []
        entities = parsed.get("entities") or {}

        # Defensive normalization — model output is untrusted shape.
        cleaned_tags = [
            t.strip().lower()
            for t in tags
            if isinstance(t, str) and t.strip()
        ][:5]  # cap at 5

        cleaned_entities: dict[str, list[str]] = {}
        for key in ("people", "organizations", "locations"):
            values = entities.get(key) or []
            cleaned_entities[key] = [
                v.strip() for v in values if isinstance(v, str) and v.strip()
            ]

        tag_extractions_total.labels(status="success").inc()
        return TagsResult(tags=cleaned_tags, entities=cleaned_entities)
