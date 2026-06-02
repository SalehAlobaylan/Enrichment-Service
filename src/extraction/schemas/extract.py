from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    url: str = Field(..., min_length=1)
    include_html: bool = False


class ExtractResponse(BaseModel):
    title: str | None = None
    text: str
    author: str | None = None
    published_at: str | None = None
    excerpt: str | None = None
    site_name: str | None = None
    image_url: str | None = None
    word_count: int
    html: str | None = None
    metadata: dict = {}


class FeedItem(BaseModel):
    title: str | None = None
    text: str = ""
    excerpt: str | None = None
    url: str | None = None
    image_url: str | None = None
    published_at: str | None = None
    author: str | None = None


class FeedExtractResponse(BaseModel):
    # is_feed False => the URL was a single article (returned as one item).
    is_feed: bool
    site_name: str | None = None
    items: list[FeedItem] = []
