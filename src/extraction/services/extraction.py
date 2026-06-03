import asyncio
import re
from html import unescape
from urllib.parse import urljoin, urlparse

from lxml import etree

from src.common.middleware.error_handler import ExtractionError
from src.common.utils.logging import get_logger
from src.common.utils.metrics import extraction_duration, extractions_total
from src.extraction.schemas.extract import (
    ExtractResponse,
    FeedExtractResponse,
    FeedItem,
)


def _get_fetcher():
    """Import Scrapling's Fetcher LAZILY (not at module top).

    Scrapling eagerly imports Playwright deep in its import chain
    (scrapling.fetchers → engines.static → toolbelt.convertor → playwright),
    and Playwright is intentionally NOT in the production image (Phase 8 — it's
    dev-only). A module-top `from scrapling.fetchers import Fetcher` therefore
    crashes the WHOLE app at startup with `ModuleNotFoundError: No module named
    'playwright'` — taking down embed/related/rerank/LLM too, including the
    embedder-only and reranker-only split deployments that never call
    /v1/extract.

    Importing here keeps the app importable everywhere; /v1/extract then fails
    at call time with a clear ExtractionError instead of killing the process.

    NOTE: this is a PERMANENT, general fix — it is NOT part of the temporary
    reranker-split workaround. It's the correct behaviour whether the service
    runs as a monolith or split, because Playwright is dev-only in production
    (Phase 8) regardless. KEEP it when the reranker split is collapsed back to
    a single deployment.
    """
    try:
        from scrapling.fetchers import Fetcher

        return Fetcher
    except Exception as exc:  # noqa: BLE001  (ModuleNotFoundError: playwright, etc.)
        raise ExtractionError(
            "Web extraction is unavailable in this deployment "
            "(Scrapling/Playwright not installed)."
        ) from exc


def _looks_like_feed(head: bytes) -> bool:
    return head.startswith(b"<?xml") or b"<rss" in head or b"<feed" in head

logger = get_logger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_IMG_SRC_RE = re.compile(r'<img[^>]+src="([^"]+)"', re.IGNORECASE)


def _strip_html(value: str | None) -> str:
    if not value:
        return ""
    return unescape(_TAG_RE.sub(" ", value)).strip()


def _first(node, selector: str):
    """First element matching a CSS selector, or None (Scrapling has no css_first)."""
    els = node.css(selector)
    return els[0] if els else None


def _meta(page, attr: str, value: str) -> str | None:
    el = _first(page, f'meta[{attr}="{value}"]')
    if el is None:
        return None
    content = el.attrib.get("content")
    return content.strip() if content else None


class ExtractionService:
    """Single-URL article/feed extractor built on Scrapling.

    Scrapling's Fetcher uses curl_cffi browser-impersonation, which gets past
    the Cloudflare/Akamai bot walls that 403 a plain HTTP client — without
    Playwright/Chromium. HTML pages are parsed with Scrapling's adaptor; RSS/Atom
    feeds are parsed from the raw body with lxml (Scrapling's own dependency,
    CDATA-safe) and the latest item is returned so pasting a feed URL autofills.
    """

    def __init__(self, timeout_sec: int = 30):
        self.timeout_sec = timeout_sec

    async def extract(self, url: str, include_html: bool = False) -> ExtractResponse:
        with extraction_duration.time():
            result = await asyncio.to_thread(self._do_extract, url, include_html)

        extractions_total.labels(status="success").inc()
        return result

    def _do_extract(self, url: str, include_html: bool) -> ExtractResponse:
        page = _get_fetcher().get(url, timeout=self.timeout_sec, stealthy_headers=True)
        raw = page.body or b""

        if _looks_like_feed(raw[:1024].lstrip().lower()):
            items, site_name = self._parse_feed(url, raw)
            if items:  # single-extract returns the latest item
                return self._feed_item_to_response(items[0], site_name)

        return self._extract_html(url, page, include_html)

    # ── Whole-feed extraction (all items) ──────────────────────
    async def extract_feed(self, url: str) -> FeedExtractResponse:
        with extraction_duration.time():
            result = await asyncio.to_thread(self._do_extract_feed, url)
        extractions_total.labels(status="success").inc()
        return result

    def _do_extract_feed(self, url: str) -> FeedExtractResponse:
        page = _get_fetcher().get(url, timeout=self.timeout_sec, stealthy_headers=True)
        raw = page.body or b""

        if _looks_like_feed(raw[:1024].lstrip().lower()):
            items, site_name = self._parse_feed(url, raw)
            if items:
                return FeedExtractResponse(
                    is_feed=True,
                    site_name=site_name,
                    items=[FeedItem(**it) for it in items],
                )

        # Not a feed — return the single article as one item.
        art = self._extract_html(url, page, include_html=False)
        return FeedExtractResponse(
            is_feed=False,
            site_name=art.site_name,
            items=[
                FeedItem(
                    title=art.title,
                    text=art.text,
                    excerpt=art.excerpt,
                    url=str(art.metadata.get("url", url)),
                    image_url=art.image_url,
                    published_at=art.published_at,
                    author=art.author,
                )
            ],
        )

    # ── RSS / Atom parsing (lxml XML, CDATA-safe) ──────────────
    def _parse_feed(self, url: str, raw: bytes) -> tuple[list[dict], str | None]:
        """Parse every <item>/<entry>. Returns (items, site_name); ([], None) if not a feed."""
        try:
            root = etree.fromstring(raw, parser=etree.XMLParser(recover=True))
        except Exception:  # noqa: BLE001
            return [], None
        if root is None:
            return [], None

        elements = root.xpath("//*[local-name()='item'] | //*[local-name()='entry']")
        if not elements:
            return [], None

        site_name = None
        site_el = root.xpath(
            "(//*[local-name()='channel']/*[local-name()='title'] "
            "| //*[local-name()='feed']/*[local-name()='title'])[1]"
        )
        if site_el and site_el[0].text:
            site_name = site_el[0].text.strip()

        items = [self._feed_item(el, url) for el in elements]
        return items, site_name

    @staticmethod
    def _feed_item(item, fallback_url: str) -> dict:
        def child(name: str):
            els = item.xpath("./*[local-name()=$n]", n=name)
            return els[0] if els else None

        def child_text(name: str) -> str | None:
            el = child(name)
            return el.text.strip() if el is not None and el.text else None

        title = child_text("title")

        # Link: RSS <link>text</link>, Atom <link href=…>, else <guid>/<id>.
        link = None
        link_el = child("link")
        if link_el is not None:
            link = link_el.get("href") or (link_el.text.strip() if link_el.text else None)
        if not link:
            link = child_text("guid") or child_text("id")
        article_url = link or fallback_url

        raw_desc = (
            child_text("description")
            or child_text("summary")
            or child_text("encoded")  # content:encoded
            or child_text("content")
        )
        description = _strip_html(raw_desc) if raw_desc else None

        published = (
            child_text("pubDate")
            or child_text("published")
            or child_text("updated")
            or child_text("date")
        )

        image = None
        enclosure = item.xpath(
            ".//*[local-name()='enclosure' or local-name()='content' "
            "or local-name()='thumbnail'][@url]"
        )
        if enclosure:
            image = enclosure[0].get("url")
        elif raw_desc:
            m = _IMG_SRC_RE.search(raw_desc)
            if m:
                image = m.group(1)
        if image:
            image = urljoin(article_url, image)

        return {
            "title": title,
            "text": description or "",
            "excerpt": (description[:300] if description else None),
            "url": article_url,
            "image_url": image,
            "published_at": published,
            "author": child_text("creator") or child_text("author"),
        }

    def _feed_item_to_response(self, it: dict, site_name: str | None) -> ExtractResponse:
        body = it["text"]
        return ExtractResponse(
            title=it["title"],
            text=body,
            author=it["author"],
            published_at=it["published_at"],
            excerpt=it["excerpt"],
            site_name=site_name or urlparse(it["url"]).netloc,
            image_url=it["image_url"],
            word_count=len(body.split()),
            html=None,
            metadata={
                "url": it["url"],
                "domain": urlparse(it["url"]).netloc,
                "source": "rss",
            },
        )

    # ── HTML article (Scrapling adaptor) ───────────────────────
    def _extract_html(self, url: str, page, include_html: bool) -> ExtractResponse:
        title = None
        title_el = _first(page, "title")
        if title_el is not None and title_el.text:
            title = title_el.text.strip()
        title = _meta(page, "property", "og:title") or title

        author = _meta(page, "name", "author")

        published = _meta(page, "property", "article:published_time") or _meta(page, "name", "date")
        if not published:
            time_el = _first(page, "time[datetime]")
            if time_el is not None:
                published = time_el.attrib.get("datetime")

        image = _meta(page, "property", "og:image") or _meta(page, "name", "twitter:image")
        if image:
            image = urljoin(url, image)

        site_name = _meta(page, "property", "og:site_name") or urlparse(url).netloc
        description = _meta(page, "name", "description") or _meta(
            page, "property", "og:description"
        )

        body_text = self._body_text(page)

        html_content = None
        if include_html:
            node = _first(page, "article") or _first(page, "main")
            if node is not None:
                html_content = node.html_content

        return ExtractResponse(
            title=title,
            text=body_text,
            author=author,
            published_at=published,
            excerpt=description,
            site_name=site_name,
            image_url=image,
            word_count=len(body_text.split()) if body_text else 0,
            html=html_content,
            metadata={"url": url, "domain": urlparse(url).netloc},
        )

    @staticmethod
    def _body_text(page) -> str:
        for selector in ["article", "main", '[role="main"]', ".post-content", ".entry-content"]:
            el = _first(page, selector)
            if el is not None:
                text = el.get_all_text()
                if text and text.strip():
                    return text.strip()

        body = _first(page, "body")
        if body is not None:
            text = body.get_all_text()
            if text:
                return text.strip()

        return ""
