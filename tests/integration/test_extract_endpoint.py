from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def test_extract_success(client: TestClient, auth_headers: dict[str, str]) -> None:
    page = MagicMock()
    page.body = (
        b"<html><title>Fixture article</title>"
        b"<article>Fixture article body</article></html>"
    )
    title = MagicMock(text="Fixture article")
    article = MagicMock()
    article.get_all_text.return_value = "Fixture article body"
    article.html_content = "<article>Fixture article body</article>"
    page.css.side_effect = lambda selector: {
        "title": [title],
        "article": [article],
    }.get(selector, [])
    fetcher = MagicMock()
    fetcher.get.return_value = page
    with patch(
        "src.extraction.services.extraction._get_fetcher", return_value=fetcher
    ):
        resp = client.post(
            "/v1/extract",
            json={"url": "https://example.com/article"},
            headers=auth_headers,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Fixture article"
    assert data["text"] == "Fixture article body"
    assert data["word_count"] == 3


def test_extract_requires_url(client: TestClient, auth_headers: dict[str, str]) -> None:
    resp = client.post(
        "/v1/extract",
        json={"url": ""},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_extract_feed_uses_real_xml_parser_with_fake_transport(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    page = MagicMock()
    page.body = b"""<?xml version="1.0"?>
    <rss><channel><title>Fixture Feed</title><item>
      <title>Arabic update</title><link>https://example.com/update</link>
      <description><![CDATA[<p>Fixture feed body</p>]]></description>
      <pubDate>Tue, 14 Jul 2026 12:00:00 GMT</pubDate>
    </item></channel></rss>"""
    fetcher = MagicMock()
    fetcher.get.return_value = page
    with patch(
        "src.extraction.services.extraction._get_fetcher", return_value=fetcher
    ):
        response = client.post(
            "/v1/extract/feed",
            json={"url": "https://example.com/feed.xml"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert response.json() == {
        "is_feed": True,
        "site_name": "Fixture Feed",
        "items": [
            {
                "title": "Arabic update",
                "text": "Fixture feed body",
                "excerpt": "Fixture feed body",
                "url": "https://example.com/update",
                "image_url": None,
                "published_at": "Tue, 14 Jul 2026 12:00:00 GMT",
                "author": None,
            }
        ],
    }


def test_extract_requires_auth(client: TestClient) -> None:
    resp = client.post("/v1/extract", json={"url": "https://example.com"})
    assert resp.status_code in (401, 403)
