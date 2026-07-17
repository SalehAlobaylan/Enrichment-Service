"""Telegram public-channel scraper (t.me/s/<username>).

Reads a channel's public web preview via Scrapling's curl_cffi fetcher — no
login, no MTProto session, no account-ban risk, server-rendered HTML (works in
the production image without Playwright). Feeds Aggregation's Source Intelligence
forward-graph: `forwarded` + `mentioned` channels are citation edges, `posts`
are the items scored for relevance, `subscribers` is promotion evidence.
"""
import asyncio
import re
from urllib.parse import urlparse

from src.common.middleware.error_handler import ExtractionError
from src.common.utils.logging import get_logger
from src.common.utils.url_guard import UnsafeURLError, validate_public_url
from src.common.workload_admission import WorkloadExecutors
from src.extraction.schemas.extract import TelegramChannelResponse, TelegramPost
from src.extraction.services.extraction import _first, _get_fetcher

logger = get_logger(__name__)

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{4,32}$")
_SUBS_RE = re.compile(r"([\d.]+)\s*([KMkm]?)")
# t.me reserved paths that look like usernames but aren't channels.
_RESERVED_PATHS = {
    "joinchat", "addstickers", "addtheme", "proxy",
    "socks", "share", "iv", "bg", "setlanguage",
}


def tg_username(raw: str) -> str:
    """Normalize a bare username, @handle, or any t.me URL to a lowercase handle."""
    s = (raw or "").strip()
    s = re.sub(r"^https?://", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^t(?:elegram)?\.me/", "", s, flags=re.IGNORECASE)
    s = s.lstrip("@")
    if s.startswith("s/"):
        s = s[2:]
    s = re.split(r"[/?#]", s, maxsplit=1)[0]
    return s.strip().lower()


def _channel_from_href(href: str, self_user: str) -> str | None:
    """First path segment of a t.me href → channel handle (excludes self/bots/`s`)."""
    if not href:
        return None
    p = urlparse(href)
    if p.netloc.lower() not in ("t.me", "telegram.me", "www.t.me"):
        return None
    seg = p.path.strip("/").split("/", 1)[0]
    u = seg.lower()
    if (
        not _USERNAME_RE.match(seg)
        or u in ("s", self_user)
        or u in _RESERVED_PATHS
        or u.endswith("bot")
    ):
        return None
    return u


def _parse_subscribers(value: str | None) -> int:
    if not value:
        return 0
    m = _SUBS_RE.search(value.replace(",", "").strip())
    if not m:
        return 0
    num = float(m.group(1))
    mult = {"k": 1_000, "m": 1_000_000}.get(m.group(2).lower(), 1)
    return int(num * mult)


class TelegramChannelService:
    """Scrape + parse one t.me/s/<username> preview page."""

    def __init__(
        self, timeout_sec: int = 30, executors: WorkloadExecutors | None = None
    ) -> None:
        self.timeout_sec = timeout_sec
        self.executors = executors

    async def fetch(self, username: str) -> TelegramChannelResponse:
        if self.executors is not None:
            return await self.executors.run("extraction", self._do_fetch, username)
        return await asyncio.to_thread(self._do_fetch, username)

    def _do_fetch(self, raw_username: str) -> TelegramChannelResponse:
        user = tg_username(raw_username)
        if not _USERNAME_RE.match(user):
            return TelegramChannelResponse(username=user, exists=False)

        url = f"https://t.me/s/{user}"
        try:
            validate_public_url(url)
        except UnsafeURLError as exc:
            raise ExtractionError(f"Refusing to fetch unsafe URL: {exc}") from exc

        page = _get_fetcher().get(url, timeout=self.timeout_sec, stealthy_headers=True)

        messages = page.css(".tgme_widget_message")
        if not messages:
            # Private channel, no public preview, or doesn't exist.
            return TelegramChannelResponse(username=user, exists=False)

        title_el = _first(page, ".tgme_channel_info_header_title")
        title = title_el.get_all_text().strip() if title_el is not None else None

        subscribers = self._subscribers(page)
        posts: list[TelegramPost] = []
        forwarded: set[str] = set()
        mentioned: set[str] = set()

        for msg in messages:
            text_el = _first(msg, ".tgme_widget_message_text")
            text = text_el.get_all_text().strip() if text_el is not None else ""

            datetime_val = None
            time_el = _first(msg, ".tgme_widget_message_date time")
            if time_el is not None:
                datetime_val = time_el.attrib.get("datetime")

            views_el = _first(msg, ".tgme_widget_message_views")
            views = views_el.get_all_text().strip() if views_el is not None else None

            if text or datetime_val:
                posts.append(TelegramPost(text=text, datetime=datetime_val, views=views))

            fwd_el = _first(msg, "a.tgme_widget_message_forwarded_from_name")
            if fwd_el is not None:
                ch = _channel_from_href(fwd_el.attrib.get("href", ""), user)
                if ch:
                    forwarded.add(ch)

            if text_el is not None:
                for a in text_el.css("a"):
                    ch = _channel_from_href(a.attrib.get("href", ""), user)
                    if ch:
                        mentioned.add(ch)

        return TelegramChannelResponse(
            username=user,
            exists=True,
            title=title,
            subscribers=subscribers,
            posts=posts,
            forwarded=sorted(forwarded),
            mentioned=sorted(mentioned),
        )

    @staticmethod
    def _subscribers(page) -> int:
        for counter in page.css(".tgme_channel_info_counter"):
            type_el = _first(counter, ".counter_type")
            if type_el is not None and type_el.get_all_text().strip().lower() == "subscribers":
                value_el = _first(counter, ".counter_value")
                if value_el is not None:
                    return _parse_subscribers(value_el.get_all_text())
        return 0
