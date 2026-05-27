"""Schemas for POST /v1/feed/news/slide — News-feed slide assembly (Slice B).

A News-feed slide is the platform's editorial unit: one featured ARTICLE
(the anchor) plus N related TWEET/COMMENT items that contextualize it.
This endpoint hands the caller the fully-assembled list — relevance from
the reranker + product-feel from the ranking rules.
"""
from pydantic import BaseModel, Field

from src.retrieval.schemas.related import RelatedItem


class FeedNewsSlideRequest(BaseModel):
    """One anchor → up to k related items."""

    anchor_content_id: str
    # Final items returned (post-ranking-rules). The slice's CLAUDE.md
    # standard is 3 related per slide; allow up to 20 for flexibility.
    k: int = Field(3, ge=1, le=20)
    # Restrict related items to specific content types. Default
    # (`["TWEET","COMMENT"]`) matches the News-feed contract; callers can
    # override (e.g., to find related ARTICLE items for a search UI).
    types: list[str] | None = None
    # Additional ids to exclude from results (anchor is auto-excluded).
    # Typical use: items the user has already been shown this session.
    exclude_ids: list[str] | None = None


class FeedNewsAnchor(BaseModel):
    """Lightweight anchor representation — caller already knows the id and
    usually has the full item from their own DB query, but having basic
    fields in the response avoids a second round trip for display purposes.
    """

    content_id: str
    type: str
    title: str | None = None
    excerpt: str | None = None
    source_name: str | None = None
    published_at: str | None = None


class FeedNewsSlideResponse(BaseModel):
    anchor: FeedNewsAnchor
    related: list[RelatedItem]
