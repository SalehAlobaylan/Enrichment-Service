import sys
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.services.extraction import ExtractionService


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

    # Patch the Fetcher at the module where it's imported
    mock_defaults = MagicMock()
    mock_defaults.Fetcher = mock_fetcher_class

    with patch.dict(sys.modules, {"scrapling.defaults": mock_defaults}):
        result = await service.extract("https://example.com/article")

    assert result.title == "Test Article"
    assert result.word_count > 0
    assert result.metadata["domain"] == "example.com"
