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


class TwitterProfileRequest(BaseModel):
    # Bare handle, @handle, or any x.com/twitter.com URL — normalized server-side.
    username: str = Field(..., min_length=1)


class TwitterPost(BaseModel):
    id: str
    text: str = ""
    created_at: str | None = None
    url: str | None = None
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    is_retweet: bool = False
    is_reply: bool = False


class TwitterProfileResponse(BaseModel):
    """Parsed public X profile timeline (syndication endpoint).

    Serves both the Source Intelligence interaction-graph (`retweeted` + `quoted`
    + `mentioned` are the citation edges, `followers` is evidence) and ingestion
    (full `posts` → TWEET content items).
    """

    username: str
    exists: bool
    rate_limited: bool = False  # syndication returned 429 (IP throttled)
    name: str | None = None
    followers: int = 0
    verified: bool = False
    image_url: str | None = None  # profile avatar
    description: str = ""         # profile bio (for source classification)
    posts: list[TwitterPost] = []
    retweeted: list[str] = []  # accounts this profile retweeted
    quoted: list[str] = []     # accounts this profile quote-tweeted
    mentioned: list[str] = []  # accounts @mentioned in tweets


class TwitterRecommendationsRequest(BaseModel):
    # Seed account: bare handle, @handle, x.com URL, or a numeric user_id —
    # normalized server-side. X recommends accounts SIMILAR to this seed.
    seed: str = Field(..., min_length=1)
    limit: int = 40


class TwitterRecAccount(BaseModel):
    """One X account X recommends as similar to the seed. All fields come inline
    from users/recommendations.json, so a candidate validates without a re-fetch.
    """

    username: str
    name: str | None = None
    followers: int = 0
    friends: int = 0
    statuses: int = 0
    listed: int = 0
    verified: bool = False
    is_protected: bool = False
    description: str = ""
    url: str | None = None         # expanded bio URL (cross-links to RSS discovery)
    image_url: str | None = None   # profile avatar (free in the API response)
    created_at: str | None = None
    user_id: str | None = None


class TwitterRecommendationsResponse(BaseModel):
    """X's "who to follow" / "قد يعجبك" graph for a seed account, via the guest-
    accessible legacy REST `users/recommendations.json` (the connect_people
    backend). Seed-relative: feeding a trusted source returns accounts X considers
    the same kind — the Source Intelligence relatedness signal.
    """

    seed: str
    exists: bool
    rate_limited: bool = False
    recommendations: list[TwitterRecAccount] = []
