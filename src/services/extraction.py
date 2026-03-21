import asyncio
from urllib.parse import urlparse

from src.schemas.extract import ExtractResponse
from src.utils.logging import get_logger
from src.utils.metrics import extraction_duration, extractions_total

logger = get_logger(__name__)


class ExtractionService:
    def __init__(self, timeout_sec: int = 30):
        self.timeout_sec = timeout_sec

    async def extract(self, url: str, include_html: bool = False) -> ExtractResponse:
        with extraction_duration.time():
            result = await asyncio.to_thread(self._do_extract, url, include_html)

        extractions_total.labels(status="success").inc()
        return result

    def _do_extract(self, url: str, include_html: bool) -> ExtractResponse:
        from scrapling.defaults import Fetcher

        fetcher = Fetcher(auto_match=True)
        page = fetcher.get(url, timeout=self.timeout_sec)

        title = None
        title_el = page.css_first("title")
        if title_el:
            title = title_el.text()

        og_title = page.css_first('meta[property="og:title"]')
        if og_title:
            title = og_title.attrib.get("content", title)

        author = None
        author_meta = page.css_first('meta[name="author"]')
        if author_meta:
            author = author_meta.attrib.get("content")

        published_at = None
        for selector in [
            'meta[property="article:published_time"]',
            'meta[name="date"]',
            "time[datetime]",
        ]:
            el = page.css_first(selector)
            if el:
                published_at = el.attrib.get("content") or el.attrib.get("datetime")
                break

        image_url = None
        og_image = page.css_first('meta[property="og:image"]')
        if og_image:
            image_url = og_image.attrib.get("content")

        site_name = None
        og_site = page.css_first('meta[property="og:site_name"]')
        if og_site:
            site_name = og_site.attrib.get("content")
        if not site_name:
            site_name = urlparse(url).netloc

        description = None
        desc_meta = page.css_first('meta[name="description"], meta[property="og:description"]')
        if desc_meta:
            description = desc_meta.attrib.get("content")

        body_text = self._extract_body_text(page)

        html_content = None
        if include_html:
            article_el = page.css_first("article") or page.css_first("main")
            if article_el:
                html_content = article_el.html

        words = body_text.split() if body_text else []

        return ExtractResponse(
            title=title,
            text=body_text,
            author=author,
            published_at=published_at,
            excerpt=description,
            site_name=site_name,
            image_url=image_url,
            word_count=len(words),
            html=html_content,
            metadata={
                "url": url,
                "domain": urlparse(url).netloc,
            },
        )

    @staticmethod
    def _extract_body_text(page) -> str:  # type: ignore[no-untyped-def]
        for selector in ["article", "main", '[role="main"]', ".post-content", ".entry-content"]:
            el = page.css_first(selector)
            if el:
                return el.text(strip=True)

        body = page.css_first("body")
        if body:
            for tag in body.css("script, style, nav, header, footer, aside"):
                tag.remove()
            return body.text(strip=True)

        return ""
