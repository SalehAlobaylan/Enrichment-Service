"""Schemas for POST /v1/feed/news/slide — News-feed slide assembly (Slice B).

This is an internal compatibility helper, not the public News-feed serving
path. It works with canonical NEWS items, filtering their ARTICLE, TWEET, and
COMMENT formats independently. CMS owns story assembly and public feed
visibility.
"""
from pydantic import BaseModel, Field

from src.retrieval.schemas.related import RelatedItem


class FeedNewsSlideRequest(BaseModel):
    """One anchor → up to k related items."""

    anchor_content_id: str
    # Final items returned (post-ranking-rules). The slice's CLAUDE.md
    # standard is 3 related per slide; allow up to 20 for flexibility.
    k: int = Field(3, ge=1, le=20)
    # Canonical kinds and optional NEWS formats are filtered independently.
    types: list[str] | None = None
    formats: list[str] | None = None
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
