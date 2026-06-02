"""Schemas for POST /v1/rerank.

⚠️ TEMP WORKAROUND (Cranl fixed-instance-RAM limit). This endpoint exists only
because the reranker had to be split onto its own deployment — see
`src/retrieval/routes/rerank.py` and `Settings.ENRICHMENT_ROLE`. When a single
instance can hold both models again, this internal endpoint can be removed.
"""
from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    """A cross-encoder rerank request: score each candidate against the query."""

    query: str
    candidates: list[str] = Field(default_factory=list)


class RerankResponse(BaseModel):
    """Relevance scores, index-aligned with the request's `candidates`."""

    scores: list[float]
    model: str
