"""Schemas for POST /v1/related."""
from pydantic import BaseModel, Field, model_validator


class RelatedRequest(BaseModel):
    """Either `content_id` (look up stored embeddings) OR `text` (embed
    fresh) must be supplied. `content_id` is the common path — News-feed
    slide assembly anchors on a content_item that already has a dense vector.
    """

    content_id: str | None = None
    text: str | None = None
    # Optional content-type filter, e.g. ["TWEET", "COMMENT"] to find related
    # social commentary for a News-feed ARTICLE anchor.
    types: list[str] | None = None
    # Final number of items to return after retrieval and optional rerank.
    # Candidate pools are sized via RELATED_K_*_DEFAULT in config; sparse
    # settings are legacy compatibility while the old CMS column still exists.
    k: int = Field(10, ge=1, le=100)
    # Caller-supplied id blocklist — typically the anchor itself + items the
    # client has already shown the user this session.
    exclude_ids: list[str] | None = None
    # When True (default), run the cross-encoder reranker on the top
    # RERANK_INPUT_K candidates and reorder by rerank score. Set
    # False for debugging or to bypass the reranker (e.g., when the model
    # isn't loaded yet during cold start).
    rerank: bool = True

    @model_validator(mode="after")
    def _need_one_anchor(self) -> "RelatedRequest":
        if not self.content_id and not self.text:
            raise ValueError("Provide either content_id or text")
        if self.content_id and self.text:
            raise ValueError("Provide content_id OR text, not both")
        return self


class RelatedItem(BaseModel):
    """One retrieved result. `sources` indicates which retrieval mode(s)
    surfaced this item — useful for debugging retrieval behavior + dashboards.

    After Slice B's rerank stage runs, `score` is the rerank score (sigmoid
    in [0, 1]) and `rerank_score` mirrors it for clarity; the original RRF
    score is overwritten. When rerank is skipped, `score` is the raw RRF
    retrieval score and `rerank_score` is None.
    """

    content_id: str
    score: float
    content_type: str | None = None
    sources: list[str]  # subset of ["dense", "sparse"]
    # Populated only when the reranker stage ran. None means: rerank was
    # disabled, or the candidate was filtered out before reaching the reranker.
    rerank_score: float | None = None
    # Surface for Slice B ranking rules (freshness decay, source diversity).
    # Both are pulled from the CMS batch-text fetch that feeds the reranker;
    # absent when rerank=False (no batch-text fetch happened).
    published_at: str | None = None
    source_name: str | None = None


class RelatedResponse(BaseModel):
    results: list[RelatedItem]
