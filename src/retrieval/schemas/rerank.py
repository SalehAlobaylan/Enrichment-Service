"""Schemas for POST /v1/rerank.

⚠️ TEMP WORKAROUND (Cranl fixed-instance-RAM limit). This endpoint exists only
because the reranker had to be split onto its own deployment — see
`src/retrieval/routes/rerank.py` and `Settings.ENRICHMENT_ROLE`. When a single
instance can hold both models again, this internal endpoint can be removed.
"""
from pydantic import BaseModel, Field, field_validator

MAX_RERANK_CANDIDATES = 32
MAX_RERANK_TEXT_CHARS = 4_000


class RerankRequest(BaseModel):
    """A cross-encoder rerank request: score each candidate against the query."""

    query: str = Field(..., min_length=1, max_length=MAX_RERANK_TEXT_CHARS)
    candidates: list[str] = Field(..., min_length=1, max_length=MAX_RERANK_CANDIDATES)

    @field_validator("candidates")
    @classmethod
    def _validate_candidates(cls, candidates: list[str]) -> list[str]:
        if any(
            not candidate.strip() or len(candidate) > MAX_RERANK_TEXT_CHARS
            for candidate in candidates
        ):
            raise ValueError("candidates must be non-empty and at most 4000 characters")
        return candidates


class RerankResponse(BaseModel):
    """Relevance scores, index-aligned with the request's `candidates`."""

    scores: list[float]
    model: str
