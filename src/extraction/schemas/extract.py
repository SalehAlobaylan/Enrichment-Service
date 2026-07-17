from pydantic import BaseModel, Field

MAX_URL_CHARS = 2_048
MAX_SOURCE_REF_CHARS = 512
MAX_SEARCH_QUERY_CHARS = 400
MAX_DISCOVERY_LIMIT = 40
MAX_RESOLVE_LINKS = 50


class ExtractRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=MAX_URL_CHARS)
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
    username: str = Field(..., min_length=1, max_length=MAX_SOURCE_REF_CHARS)


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
    username: str = Field(..., min_length=1, max_length=MAX_SOURCE_REF_CHARS)


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
    seed: str = Field(..., min_length=1, max_length=MAX_SOURCE_REF_CHARS)
    limit: int = Field(default=40, ge=1, le=MAX_DISCOVERY_LIMIT)


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


# ---------- YouTube (InnerTube / youtubei) ----------


class YouTubeChannelRequest(BaseModel):
    # @handle, UC… channel id, or a youtube.com channel URL.
    channel: str = Field(..., min_length=1, max_length=MAX_SOURCE_REF_CHARS)


class YouTubeVideo(BaseModel):
    video_id: str
    title: str = ""


class YouTubeChannelResponse(BaseModel):
    """A YouTube channel read via the guest InnerTube WEB client (no API key, no
    quota). Recent video titles are the text CMS scores for relevance; subscribers
    feed the authority signal.
    """

    channel: str
    exists: bool
    channel_id: str | None = None
    title: str | None = None
    subscribers: int = 0
    description: str = ""
    image_url: str | None = None  # channel avatar
    videos: list[YouTubeVideo] = []
    # Audio-first detection (from a sample of recent videos' YouTube category):
    # Music/Gaming/Sports/Film need the picture; News/Education/Entertainment/etc.
    # are talk-driven and work audio-first. category = the dominant sampled one.
    category: str | None = None
    audio_first: bool = True
    top_duration_sec: int = 0  # newest video length (also feeds the long-form guard)


class YouTubeRelatedRequest(BaseModel):
    channel: str = Field(..., min_length=1, max_length=MAX_SOURCE_REF_CHARS)


class YouTubeRelatedChannel(BaseModel):
    channel_id: str
    name: str | None = None
    via: str = "youtube-watchnext"


class YouTubeRelatedResponse(BaseModel):
    """Channels surfaced in the watch-next graph of a seed channel's recent video
    — the YouTube analog of forwards/retweets. Each is a candidate the graph
    scores + promotes.
    """

    channel: str
    exists: bool
    related: list[YouTubeRelatedChannel] = []


# ---------- Apple Podcasts "Listeners Also Subscribed" ----------


class ApplePodcastRelatedRequest(BaseModel):
    # Apple podcast collection id (numeric adamId).
    collection_id: str = Field(..., min_length=1, max_length=64)
    country: str = Field(default="us", min_length=2, max_length=2)


class AppleRelatedShow(BaseModel):
    adam_id: str
    title: str | None = None
    genres: list[str] = []


class ApplePodcastRelatedResponse(BaseModel):
    """Shows in a seed podcast's "You Might Also Like" / "Listeners Also
    Subscribed" shelf — Apple's co-listen relation, scraped from the public show
    page's embedded JSON (no token, no login). Each adam_id resolves to an RSS
    feed via the iTunes lookup API on the Aggregation side.
    """

    collection_id: str
    exists: bool
    related: list[AppleRelatedShow] = []


# ---------- YouTube channel search (topic/keyword discovery) ----------


class YouTubeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_SEARCH_QUERY_CHARS)
    limit: int = Field(default=15, ge=1, le=MAX_DISCOVERY_LIMIT)


class YouTubeSearchChannel(BaseModel):
    channel_id: str
    title: str | None = None
    subscribers: int = 0


class YouTubeSearchResponse(BaseModel):
    """Channels matching a topic query via guest InnerTube search (channels
    filter). The YouTube analog of the iTunes podcast keyword net — feeds
    interest-driven discovery for media profiles."""

    query: str
    channels: list[YouTubeSearchChannel] = []


# ---------- YouTube podcast extraction (podcast-intent search + feed paste) ----------


class ExtractedChannel(BaseModel):
    """A channel pulled out of an InnerTube payload (search results or a pasted
    youtubei/v1 feed). `is_podcast` is True when YouTube tagged the source lockup
    LOCKUP_CONTENT_TYPE_PODCAST (or it owns a podcast playlist) — the explicit
    podcast designation we use to surface إذاعة ثمانية / Mics مايكس etc. Subscribers/
    audio-first are filled later by the per-channel enrichment (fetch_channel)."""

    channel_id: str
    title: str | None = None
    handle: str | None = None  # @handle (from canonicalBaseUrl)
    is_podcast: bool = False
    episode_count: int = 0
    subscribers: int = 0
    mention_count: int = 0  # times referenced in the payload (relevance proxy)


class YouTubePodcastSearchRequest(BaseModel):
    # Podcast-intent phrase, e.g. "بودكاست اقتصاد" / "tech podcast".
    query: str = Field(..., min_length=1, max_length=MAX_SEARCH_QUERY_CHARS)
    limit: int = Field(default=15, ge=1, le=MAX_DISCOVERY_LIMIT)


class YouTubePodcastSearchResponse(BaseModel):
    """Channels behind the podcast results of a podcast-intent guest search — the
    fix for the channel-only/raw-keyword search that never surfaced YouTube's
    podcast shelves. Podcast-tagged channels are returned first."""

    query: str
    channels: list[ExtractedChannel] = []


class YouTubeParseFeedRequest(BaseModel):
    # A raw youtubei/v1 response object (e.g. a personalized home feed the admin
    # pasted). Parsed server-side into the channels it references.
    raw: dict


class YouTubeResolveLinksRequest(BaseModel):
    # One YouTube reference per item: a @handle, channel URL, /channel/UC… URL,
    # raw UC… id, or any video / share / shorts link. The low-friction seed path
    # (each line is two taps off the Share button — no DevTools JSON paste).
    inputs: list[str] = Field(
        default_factory=list, min_length=1, max_length=MAX_RESOLVE_LINKS
    )


class YouTubeResolveLinksResponse(BaseModel):
    """Distinct channels behind pasted YouTube links/handles. Channel refs resolve
    directly; video refs resolve to their owning channel. Each is enriched + queued
    for review on the Aggregation side (guest InnerTube — no auth, no quota)."""

    channels: list[ExtractedChannel] = []


class YouTubeParseFeedResponse(BaseModel):
    """Distinct channels referenced by a pasted youtubei/v1 payload — the manual
    seed path. Each channel is then enriched + queued for review on the
    Aggregation side (no stored credentials; the admin stays the session)."""

    channels: list[ExtractedChannel] = []
