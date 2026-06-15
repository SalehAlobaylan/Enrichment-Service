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


class TelegramChannelRequest(BaseModel):
    # Bare username, @handle, or any t.me URL — normalized server-side.
    username: str = Field(..., min_length=1)


class TelegramPost(BaseModel):
    text: str = ""
    datetime: str | None = None
    views: str | None = None


class TelegramChannelResponse(BaseModel):
    """Parsed public preview of a Telegram channel (t.me/s/<username>).

    Used by Aggregation's Source Intelligence forward-graph: `forwarded` +
    `mentioned` are the citation edges, `posts` are the items scored for
    relevance, `subscribers` is promotion evidence.
    """

    username: str
    exists: bool
    title: str | None = None
    subscribers: int = 0
    posts: list[TelegramPost] = []
    forwarded: list[str] = []  # channels this channel forwarded from
    mentioned: list[str] = []  # channels linked via t.me/ in post text
