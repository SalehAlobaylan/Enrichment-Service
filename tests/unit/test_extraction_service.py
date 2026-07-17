from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.common.middleware.error_handler import ExtractionError
from src.common.utils.url_guard import UnsafeURLError, safe_url_label
from src.extraction.services.extraction import (
    MAX_FEED_ITEMS,
    MAX_RESPONSE_BYTES,
    ExtractionService,
    _BoundedFetcher,
)


@pytest.fixture
def service() -> ExtractionService:
    return ExtractionService(timeout_sec=10)


@pytest.mark.asyncio
async def test_extract_returns_response(service: ExtractionService) -> None:
    mock_page = MagicMock()
    mock_page.body = (
        b"<html><title>Test Article</title>"
        b"<article>Article body text here</article></html>"
    )

    mock_title = MagicMock()
    mock_title.text = "Test Article"

    mock_body = MagicMock()
    mock_body.get_all_text.return_value = "Article body text here with some words"

    mock_article = MagicMock()
    mock_article.get_all_text.return_value = "Article body text here with some words"
    mock_article.html = "<article>Article body text here with some words</article>"

    def css_handler(sel: str) -> list[MagicMock]:
        mapping: dict[str, list[MagicMock]] = {
            "title": [mock_title],
            "article": [mock_article],
            "body": [mock_body],
        }
        return mapping.get(sel, [])

    mock_page.css = MagicMock(side_effect=css_handler)

    mock_fetcher_class = MagicMock()
    mock_fetcher_class.get.return_value = mock_page

    with patch("src.extraction.services.extraction._get_fetcher", return_value=mock_fetcher_class):
        result = await service.extract("https://example.com/article")

    assert result.title == "Test Article"
    assert result.word_count > 0
    assert result.metadata["domain"] == "example.com"


def test_extraction_rejects_oversized_response_before_parsing(
    service: ExtractionService,
) -> None:
    page = MagicMock(body=b"x" * (MAX_RESPONSE_BYTES + 1), status=200)
    fetcher = MagicMock()
    fetcher.get.return_value = page
    with (
        patch("src.extraction.services.extraction._get_fetcher", return_value=fetcher),
        patch("src.extraction.services.extraction.validate_public_url"),
        pytest.raises(ExtractionError, match="exceeded size limit"),
    ):
        service._do_extract("https://public.test/article", include_html=False)
    page.css.assert_not_called()


def test_bounded_transport_aborts_during_write_callback_before_selector_creation() -> None:
    def receive_oversized(*_args, **kwargs):
        kwargs["content_callback"](b"x" * (MAX_RESPONSE_BYTES + 1))
        raise RuntimeError("curl write callback aborted")

    with (
        patch(
            "src.extraction.services.extraction.cffi_requests.get",
            side_effect=receive_oversized,
        ),
        pytest.raises(ExtractionError, match="exceeded size limit"),
    ):
        _BoundedFetcher().get("https://public.test/article", timeout=1)


def test_bounded_transport_parses_only_callback_bytes() -> None:
    response = MagicMock(status_code=200, headers={"content-type": "text/html"})

    def receive_fixture(*_args, **kwargs):
        kwargs["content_callback"](b"<html><title>Bounded fixture</title></html>")
        return response

    with patch(
        "src.extraction.services.extraction.cffi_requests.get",
        side_effect=receive_fixture,
    ):
        page = _BoundedFetcher().get("https://public.test/article", timeout=1)

    assert page.body == b"<html><title>Bounded fixture</title></html>"
    assert page.css("title")[0].text == "Bounded fixture"


def test_extraction_rejects_unsupported_content_type_before_parsing(
    service: ExtractionService,
) -> None:
    page = MagicMock(body=b"binary", status=200, headers={"content-type": "image/png"})
    fetcher = MagicMock()
    fetcher.get.return_value = page
    with (
        patch("src.extraction.services.extraction._get_fetcher", return_value=fetcher),
        patch("src.extraction.services.extraction.validate_public_url"),
        pytest.raises(ExtractionError, match="unsupported content type"),
    ):
        service._do_extract("https://public.test/image", include_html=False)
    page.css.assert_not_called()


def test_feed_parser_caps_item_materialization(service: ExtractionService) -> None:
    items = "".join(
        f"<item><title>{index}</title><description>text</description></item>"
        for index in range(MAX_FEED_ITEMS + 1)
    )
    parsed, _ = service._parse_feed(
        "https://public.test/feed.xml", f"<rss><channel>{items}</channel></rss>".encode()
    )
    assert len(parsed) == MAX_FEED_ITEMS


def test_unsafe_url_error_does_not_expose_resolver_detail(service: ExtractionService) -> None:
    with (
        patch(
            "src.extraction.services.extraction.validate_public_url",
            side_effect=UnsafeURLError("private host 10.0.0.8 secret-token"),
        ),
        pytest.raises(ExtractionError, match="Extraction URL is not permitted") as error,
    ):
        service._do_extract("https://private.test/?token=secret-token", include_html=False)
    assert "secret-token" not in str(error.value)


def test_safe_url_label_redacts_userinfo_query_and_fragment() -> None:
    label = safe_url_label("https://user:secret@example.test/path?token=secret#fragment")
    assert label.startswith("https://example.test/#")
    assert "secret" not in label


def test_redirect_destination_is_validated_before_second_fetch(
    service: ExtractionService,
) -> None:
    redirect = SimpleNamespace(
        status=302,
        headers={"location": "https://private.test/next"},
    )
    fetcher = MagicMock()
    fetcher.get.return_value = redirect
    with (
        patch("src.extraction.services.extraction._get_fetcher", return_value=fetcher),
        patch(
            "src.extraction.services.extraction.validate_public_url",
            side_effect=[None, UnsafeURLError("private redirect")],
        ) as validate,
        pytest.raises(ExtractionError, match="Extraction URL is not permitted"),
    ):
        service._fetch_checked("https://public.test/start")

    assert validate.call_count == 2
    assert fetcher.get.call_count == 1
