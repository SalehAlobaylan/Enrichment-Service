"""YouTube channel reader via the guest InnerTube (youtubei/v1) WEB client.

The official Data API is a dead end for discovery: it deprecated featured
channels (channelSections) and a `search` call costs 100 quota. InnerTube — the
same internal API yt-dlp / Invidious use — needs **no API key and burns no
quota**, via the public WEB client context (key embedded in youtube.com JS, same
for everyone). PO tokens only gate playback/streaming, not the search/browse/next
metadata we read here. Live-probed 2026-06: search→channelRenderer, browse Videos
tab→lockupViewModel, next→secondaryResults lockupViewModel (related channels).

Serves the media Source Intelligence graph: recent video titles (relevance text)
+ subscribers (authority) per channel, and the watch-next channel↔channel edges.
All requests go through curl_cffi browser-impersonation; failures degrade to
`exists=False` (graceful — never crashes the graph build).
"""

import asyncio
import json
import os
import re

from curl_cffi import requests as cffi

from src.common.utils.logging import get_logger
from src.extraction.schemas.extract import (
    ExtractedChannel,
    YouTubeChannelResponse,
    YouTubePodcastSearchResponse,
    YouTubeRelatedChannel,
    YouTubeRelatedResponse,
    YouTubeSearchChannel,
    YouTubeSearchResponse,
    YouTubeVideo,
)

logger = get_logger(__name__)

# Public WEB InnerTube key (embedded in youtube.com JS — not a secret). Env-overridable.
_KEY = os.getenv("YOUTUBE_INNERTUBE_KEY", "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8")
_CLIENT_VERSION = os.getenv("YOUTUBE_INNERTUBE_CLIENT_VERSION", "2.20260601.01.00")
_BASE = "https://www.youtube.com/youtubei/v1"
# Base64 params for a channel's "Videos" tab.
_VIDEOS_TAB_PARAMS = "EgZ2aWRlb3PyBgQKAjoA"
# Base64 params for the search "channels-only" filter.
_SEARCH_CHANNELS_PARAMS = "EgIQAg%3D%3D"

# YouTube categories whose content needs the picture (not audio-first). Everything
# else (News & Politics, Education, Entertainment, People & Blogs, Comedy, Science
# & Technology, Howto …) is talk-driven and works as audio-first For You content.
_VISUAL_CATEGORIES = {"Music", "Gaming", "Sports", "Film & Animation"}

_UC_RE = re.compile(r"^UC[0-9A-Za-z_-]{22}$")
_UC_ANY = re.compile(r"UC[0-9A-Za-z_-]{22}")
_SUBS_RE = re.compile(
    r"([\d.,]+)\s*(مليار|مليون|ألف|الف|[KMBkmb])?\s*(?:مشترك|مشتركاً|subscriber)",
)
# Episode-count badge on a podcast lockup ("147 حلقة" / "12 episodes").
_EP_RE = re.compile(r"(\d[\d,]*)\s*(?:حلقة|حلقات|episodes?)")
# videoId out of a watch / youtu.be / shorts / embed link.
_VID_RE = re.compile(
    r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([0-9A-Za-z_-]{11})"
)


def _context(hl: str = "ar", gl: str = "SA") -> dict:
    return {"client": {"clientName": "WEB", "clientVersion": _CLIENT_VERSION, "hl": hl, "gl": gl}}


def _post(endpoint: str, body: dict, timeout: int) -> dict:
    r = cffi.post(
        f"{_BASE}/{endpoint}?key={_KEY}&prettyPrint=false",
        json={"context": _context(), **body},
        headers={"Content-Type": "application/json"},
        impersonate="chrome",
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def _find_first(obj, key, depth: int = 0):
    if depth > 22:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                return v
            r = _find_first(v, key, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_first(v, key, depth + 1)
            if r is not None:
                return r
    return None


def _find_all(obj, key, out, depth: int = 0):
    if depth > 22:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            else:
                _find_all(v, key, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _find_all(v, key, out, depth + 1)


def _scan_subscribers(obj, depth: int = 0) -> int:
    """Walk all string values and return the first parseable subscriber count.
    Robust to the view-model nesting (the count lives several levels deep under a
    `content` key that is itself inside another `content`)."""
    if depth > 26:
        return 0
    if isinstance(obj, str):
        if "مشترك" in obj or "subscriber" in obj.lower():
            return _parse_subscribers(obj)
        return 0
    if isinstance(obj, dict):
        for v in obj.values():
            r = _scan_subscribers(v, depth + 1)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _scan_subscribers(v, depth + 1)
            if r:
                return r
    return 0


def _parse_subscribers(blob: str) -> int:
    m = _SUBS_RE.search(blob)
    if not m:
        return 0
    num = m.group(1).replace(",", "")
    try:
        val = float(num)
    except ValueError:
        return 0
    suffix = (m.group(2) or "").lower()
    mult = {
        "مليار": 1_000_000_000,
        "مليون": 1_000_000,
        "ألف": 1_000,
        "الف": 1_000,
        "k": 1_000,
        "m": 1_000_000,
        "b": 1_000_000_000,
    }.get(suffix, 1)
    return int(val * mult)


def _channel_ref(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"^https?://", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(www\.)?youtube\.com/", "", s, flags=re.IGNORECASE)
    for prefix in ("channel/", "c/", "user/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return re.split(r"[/?#]", s, maxsplit=1)[0].strip()


def _channel_title_for(lv: dict, cid: str) -> str | None:
    """The display name of channel `cid` as it appears in a lockup's metadata
    rows (the text run whose tap targets that channel's browse endpoint)."""
    rowsets: list = []
    _find_all(lv, "metadataRows", rowsets)
    for rowset in rowsets:
        if not isinstance(rowset, list):
            continue
        for row in rowset:
            for part in (row or {}).get("metadataParts") or []:
                txt = part.get("text") or {}
                content = txt.get("content")
                if not content:
                    continue
                for run in txt.get("commandRuns") or []:
                    be = (
                        ((run.get("onTap") or {}).get("innertubeCommand") or {})
                        .get("browseEndpoint")
                        or {}
                    )
                    if be.get("browseId") == cid:
                        return content.strip()
    # Fallback: the avatar a11y label ("الانتقال إلى قناة \"إذاعة مختلف\"").
    a11y = _find_first(lv, "a11yLabel")
    if isinstance(a11y, str):
        m = re.search(r'["“](.+?)["”]', a11y)
        if m:
            return m.group(1).strip()
    return None


def _episode_count(lv: dict) -> int:
    badges: list = []
    _find_all(lv, "thumbnailBadgeViewModel", badges)
    for b in badges:
        t = b.get("text")
        if isinstance(t, str):
            m = _EP_RE.search(t)
            if m:
                try:
                    return int(m.group(1).replace(",", ""))
                except ValueError:
                    return 0
    return 0


def _channel_id_in(node: dict) -> tuple[str | None, str | None]:
    """First channel (UC…) browse endpoint inside a subtree → (channel_id, @handle)."""
    bes: list = []
    _find_all(node, "browseEndpoint", bes)
    for be in bes:
        bid = be.get("browseId") or ""
        if _UC_RE.match(bid):
            cbu = be.get("canonicalBaseUrl") or ""
            handle = cbu[1:] if cbu.startswith("/@") else None
            return bid, handle
    return None, None


def _video_owner(vr: dict) -> tuple[str | None, str | None, str | None]:
    """The publishing channel of a search videoRenderer → (channel_id, @handle, name).
    Search returns videos, not channels/lockups — but a podcast-intent query's video
    owners are the podcast networks (إذاعة ثمانية, Mics مايكس …)."""
    by = vr.get("ownerText") or vr.get("longBylineText") or vr.get("shortBylineText") or {}
    for run in by.get("runs") or []:
        nav = (run.get("navigationEndpoint") or {}).get("browseEndpoint") or {}
        bid = nav.get("browseId") or ""
        if _UC_RE.match(bid):
            cbu = nav.get("canonicalBaseUrl") or ""
            handle = cbu[1:] if cbu.startswith("/@") else None
            return bid, handle, (run.get("text") or "").strip() or None
    return None, None, None


def _extract_channels_from_lockups(data: dict) -> list[ExtractedChannel]:
    """The shared core for podcast-intent search AND pasted-feed import: pull the
    distinct channels an InnerTube payload references. Reads channel-publisher
    links out of `lockupViewModel`s (podcast shelves + video lockups carry their
    publishing channel's browseEndpoint) and plain `channelRenderer`s, marking
    podcast-tagged channels. Subscribers/audio-first are enriched downstream."""
    out: dict[str, ExtractedChannel] = {}

    lockups: list = []
    _find_all(data, "lockupViewModel", lockups)
    for lv in lockups:
        if not isinstance(lv, dict):
            continue
        is_pod = lv.get("contentType") == "LOCKUP_CONTENT_TYPE_PODCAST"
        cid, handle = _channel_id_in(lv)
        if not cid:
            continue
        episodes = _episode_count(lv) if is_pod else 0
        existing = out.get(cid)
        if existing:
            existing.mention_count += 1
            if is_pod:
                existing.is_podcast = True
            if episodes and not existing.episode_count:
                existing.episode_count = episodes
            if handle and not existing.handle:
                existing.handle = handle
            continue
        out[cid] = ExtractedChannel(
            channel_id=cid,
            title=_channel_title_for(lv, cid),
            handle=handle,
            is_podcast=is_pod,
            episode_count=episodes,
            mention_count=1,
        )

    # Plain channel results (the channels-filter search shape).
    crs: list = []
    _find_all(data, "channelRenderer", crs)
    for cr in crs:
        cid = cr.get("channelId")
        if not cid:
            continue
        ex = out.get(cid)
        if ex:
            ex.mention_count += 1
            continue
        out[cid] = ExtractedChannel(
            channel_id=cid,
            title=(cr.get("title") or {}).get("simpleText"),
            subscribers=_scan_subscribers(cr),
            mention_count=1,
        )

    # Video results: a plain web search returns videoRenderers, whose owners are
    # the publishing channels. For a podcast-intent query those owners are the
    # podcast networks — and how often a channel owns the top results is a strong
    # relevance signal, so accumulate it as mention_count.
    vrs: list = []
    _find_all(data, "videoRenderer", vrs)
    for vr in vrs:
        cid, handle, name = _video_owner(vr)
        if not cid:
            continue
        ex = out.get(cid)
        if ex:
            ex.mention_count += 1
            if handle and not ex.handle:
                ex.handle = handle
            if name and not ex.title:
                ex.title = name
            continue
        out[cid] = ExtractedChannel(
            channel_id=cid, title=name, handle=handle, mention_count=1
        )
    return list(out.values())


class YouTubeInnerTubeService:
    """Guest InnerTube reader — channel videos/subscribers + watch-next relations."""

    def __init__(self, timeout_sec: int = 30):
        self.timeout_sec = timeout_sec

    async def fetch_channel(self, channel: str) -> YouTubeChannelResponse:
        return await asyncio.to_thread(self._do_fetch_channel, channel)

    async def fetch_related(self, channel: str) -> YouTubeRelatedResponse:
        return await asyncio.to_thread(self._do_fetch_related, channel)

    async def search_channels(self, query: str, limit: int = 15) -> YouTubeSearchResponse:
        return await asyncio.to_thread(self._do_search_channels, query, limit)

    async def search_podcasts(
        self, query: str, limit: int = 15
    ) -> YouTubePodcastSearchResponse:
        return await asyncio.to_thread(self._do_search_podcasts, query, limit)

    async def parse_feed(self, raw: dict) -> list[ExtractedChannel]:
        return await asyncio.to_thread(self._do_parse_feed, raw)

    async def resolve_links(self, inputs: list[str]) -> list[ExtractedChannel]:
        return await asyncio.to_thread(self._do_resolve_links, inputs)

    # ---- internals (sync; run in a thread) ----

    def _do_search_podcasts(self, query: str, limit: int) -> YouTubePodcastSearchResponse:
        q = (query or "").strip()
        if not q:
            return YouTubePodcastSearchResponse(query=query, channels=[])
        try:
            # No channels-only filter: the podcast-intent query itself biases the
            # results, and we want the podcast shelves (lockupViewModel) too — that
            # is where إذاعة ثمانية / Mics مايكس surface, not channelRenderer.
            data = _post("search", {"query": q}, self.timeout_sec)
        except Exception as exc:
            logger.debug("youtube_podcast_search_failed", query=q, error=str(exc))
            return YouTubePodcastSearchResponse(query=q, channels=[])
        chans = _extract_channels_from_lockups(data)
        # Podcast-tagged channels first, then the most-referenced owners (a channel
        # that owns several of the top results for "بودكاست <topic>" is the strongest
        # candidate).
        chans.sort(key=lambda c: (0 if c.is_podcast else 1, -c.mention_count))
        return YouTubePodcastSearchResponse(query=q, channels=chans[: max(1, limit)])

    def _do_parse_feed(self, raw: dict) -> list[ExtractedChannel]:
        if not isinstance(raw, dict):
            return []
        return _extract_channels_from_lockups(raw)

    def _do_search_channels(self, query: str, limit: int) -> YouTubeSearchResponse:
        q = (query or "").strip()
        if not q:
            return YouTubeSearchResponse(query=query, channels=[])
        try:
            data = _post(
                "search", {"query": q, "params": _SEARCH_CHANNELS_PARAMS}, self.timeout_sec
            )
        except Exception as exc:
            logger.debug("youtube_search_failed", query=q, error=str(exc))
            return YouTubeSearchResponse(query=q, channels=[])
        renderers: list = []
        _find_all(data, "channelRenderer", renderers)
        out: list[YouTubeSearchChannel] = []
        seen: set[str] = set()
        for cr in renderers:
            cid = cr.get("channelId")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            title = (cr.get("title") or {}).get("simpleText")
            # In search results YouTube swaps the labels: the subscriber count
            # ("341 ألف مشترك") lands in `videoCountText` while `subscriberCountText`
            # holds the @handle. Scan the whole renderer for the subscriber string.
            subs = _scan_subscribers(cr)
            out.append(YouTubeSearchChannel(channel_id=cid, title=title, subscribers=subs))
            if len(out) >= max(1, limit):
                break
        return YouTubeSearchResponse(query=q, channels=out)

    def _do_resolve_links(self, inputs: list[str]) -> list[ExtractedChannel]:
        """Turn pasted YouTube references (a @handle, channel URL, or any video /
        share / shorts link) into the channels behind them — guest-side, no auth,
        no quota. This is the low-friction alternative to pasting a 1 MB youtubei
        feed: every line is two taps off YouTube's Share button. A channel ref
        resolves directly; a video ref resolves to its owning channel."""
        out: list[ExtractedChannel] = []
        seen: set[str] = set()
        for raw in inputs or []:
            chan = self._resolve_one(raw)
            if chan and chan.channel_id not in seen:
                seen.add(chan.channel_id)
                out.append(chan)
        return out

    def _resolve_one(self, raw: str) -> ExtractedChannel | None:
        s = (raw or "").strip()
        if not s:
            return None
        # Raw channel id or /channel/UC… URL — use it directly, no round-trip.
        m = _UC_ANY.search(s)
        if m and ("/channel/" in s or _UC_RE.match(s)):
            return ExtractedChannel(channel_id=m.group(0))
        # A bare video id pasted on its own.
        if re.fullmatch(r"[0-9A-Za-z_-]{11}", s) and not s.startswith("@"):
            return self._channel_from_video(s)
        # Normalize to a URL InnerTube can resolve.
        if s.startswith("http://") or s.startswith("https://"):
            url = s
        elif s.startswith("@"):
            url = f"https://www.youtube.com/{s}"
        else:
            url = f"https://www.youtube.com/@{s.lstrip('@')}"
        try:
            data = _post("navigation/resolve_url", {"url": url}, self.timeout_sec)
        except Exception as exc:
            logger.debug("youtube_resolve_link_failed", ref=s, error=str(exc))
            return None
        ep = data.get("endpoint", data)
        browse = _find_first(ep, "browseEndpoint")
        if isinstance(browse, dict):
            bid = browse.get("browseId") or ""
            if _UC_RE.match(bid):
                base = browse.get("canonicalBaseUrl") or ""
                handle = base.lstrip("/") if base.startswith("/@") else None
                return ExtractedChannel(channel_id=bid, handle=handle)
        # Not a channel — pull the videoId and resolve its owner.
        watch = _find_first(ep, "watchEndpoint")
        vid = watch.get("videoId") if isinstance(watch, dict) else None
        if not vid:
            vm = _VID_RE.search(json.dumps(ep))
            vid = vm.group(1) if vm else None
        return self._channel_from_video(vid) if vid else None

    def _channel_from_video(self, video_id: str) -> ExtractedChannel | None:
        try:
            data = _post("player", {"videoId": video_id}, self.timeout_sec)
        except Exception as exc:
            logger.debug("youtube_player_failed", video=video_id, error=str(exc))
            return None
        vd = data.get("videoDetails") or {}
        cid = vd.get("channelId")
        if not cid or not _UC_RE.match(cid):
            return None
        return ExtractedChannel(channel_id=cid, title=vd.get("author") or None)

    def _resolve_channel_id(self, channel: str) -> str | None:
        ref = _channel_ref(channel)
        if _UC_RE.match(ref):
            return ref
        # Resolve a handle / custom name via search (channels filter).
        try:
            data = _post(
                "search",
                {"query": ref.lstrip("@"), "params": _SEARCH_CHANNELS_PARAMS},
                self.timeout_sec,
            )
        except Exception as exc:
            logger.debug("youtube_resolve_failed", channel=channel, error=str(exc))
            return None
        cr = _find_first(data, "channelRenderer")
        if cr and cr.get("channelId"):
            return cr["channelId"]
        return None

    def _do_fetch_channel(self, channel: str) -> YouTubeChannelResponse:
        cid = self._resolve_channel_id(channel)
        if not cid:
            return YouTubeChannelResponse(channel=channel, exists=False)
        try:
            data = _post(
                "browse", {"browseId": cid, "params": _VIDEOS_TAB_PARAMS}, self.timeout_sec
            )
        except Exception as exc:
            logger.debug("youtube_browse_failed", channel=channel, error=str(exc))
            return YouTubeChannelResponse(channel=channel, exists=False)

        meta = _find_first(data, "channelMetadataRenderer") or {}
        title = meta.get("title")
        description = meta.get("description") or ""
        avatars = (meta.get("avatar") or {}).get("thumbnails") or []
        image_url = avatars[-1]["url"] if avatars else None

        videos: list[YouTubeVideo] = []
        lockups: list = []
        _find_all(data, "lockupViewModel", lockups)
        for lv in lockups:
            if lv.get("contentType") != "LOCKUP_CONTENT_TYPE_VIDEO":
                continue
            vid = lv.get("contentId")
            t = (
                (lv.get("metadata") or {})
                .get("lockupMetadataViewModel", {})
                .get("title", {})
                .get("content", "")
            )
            if vid:
                videos.append(YouTubeVideo(video_id=vid, title=(t or "").strip()))

        # Subscriber count lives deep in the page-header view-model; scan all
        # strings in the header subtree for the first parseable count.
        subs = _scan_subscribers(data.get("header") or {}) or _scan_subscribers(
            data.get("metadata") or {}
        )

        # Audio-first detection: sample a few recent videos' YouTube category via
        # /player (no PO token needed for category/duration). Majority-visual →
        # not audio-first. top_duration also feeds the long-form guard.
        category, audio_first, top_duration = self._classify_audio_first(videos)

        return YouTubeChannelResponse(
            channel=channel,
            exists=True,
            channel_id=cid,
            title=(title or "").strip() or None,
            subscribers=subs,
            description=description.strip(),
            image_url=image_url,
            videos=videos[:15],
            category=category,
            audio_first=audio_first,
            top_duration_sec=top_duration,
        )

    def _video_category_duration(self, video_id: str) -> tuple[str | None, int]:
        try:
            d = _post("player", {"videoId": video_id}, self.timeout_sec)
        except Exception:
            return None, 0
        mf = (d.get("microformat") or {}).get("playerMicroformatRenderer") or {}
        vd = d.get("videoDetails") or {}
        try:
            dur = int(vd.get("lengthSeconds") or 0)
        except (TypeError, ValueError):
            dur = 0
        return mf.get("category"), dur

    def _classify_audio_first(self, videos: list[YouTubeVideo]) -> tuple[str | None, bool, int]:
        # Sample the two newest videos' categories. Flag NOT audio-first only when
        # ALL sampled categories are visual (Music/Gaming/Sports/Film) — a single
        # off or mis-tagged upload (e.g. a business channel whose latest video is
        # tagged Music) must not misflag an otherwise talk channel. top_duration =
        # newest video length (also feeds the long-form guard).
        cats: list[str] = []
        durs: list[int] = []
        for v in videos[:2]:
            cat, dur = self._video_category_duration(v.video_id)
            if cat:
                cats.append(cat)
            if dur:
                durs.append(dur)
        top_duration = durs[0] if durs else 0
        if not cats:
            return None, True, top_duration  # unknown → don't penalize
        dominant = next((c for c in cats if c in _VISUAL_CATEGORIES), cats[0])
        audio_first = not all(c in _VISUAL_CATEGORIES for c in cats)
        return dominant, audio_first, top_duration

    def _do_fetch_related(self, channel: str) -> YouTubeRelatedResponse:
        ch = self._do_fetch_channel(channel)
        if not ch.exists or not ch.videos:
            return YouTubeRelatedResponse(channel=channel, exists=ch.exists)
        try:
            data = _post("next", {"videoId": ch.videos[0].video_id}, self.timeout_sec)
        except Exception as exc:
            logger.debug("youtube_next_failed", channel=channel, error=str(exc))
            return YouTubeRelatedResponse(channel=channel, exists=True)

        sec = _find_first(data, "secondaryResults") or {}
        lockups: list = []
        _find_all(sec, "lockupViewModel", lockups)
        seen: set[str] = set()
        related: list[YouTubeRelatedChannel] = []
        for lv in lockups:
            blob = str(lv)
            for uc in _UC_ANY.findall(blob):
                if uc == ch.channel_id or uc in seen:
                    continue
                seen.add(uc)
                related.append(YouTubeRelatedChannel(channel_id=uc, via="youtube-watchnext"))
        return YouTubeRelatedResponse(channel=channel, exists=True, related=related[:25])
