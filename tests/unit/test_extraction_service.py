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

    mock_title = MagicMock()
    mock_title.text.return_value = "Test Article"

    mock_body = MagicMock()
    mock_body.css.return_value = []
    mock_body.text.return_value = "Article body text here with some words"

    mock_article = MagicMock()
    mock_article.text.return_value = "Article body text here with some words"
    mock_article.html = "<article>Article body text here with some words</article>"

    def css_first_handler(sel: str) -> MagicMock | None:
        mapping: dict[str, MagicMock | None] = {
            "title": mock_title,
            "article": mock_article,
            "body": mock_body,
        }
        return mapping.get(sel)

    mock_page.css_first = MagicMock(side_effect=css_first_handler)

    mock_fetcher_instance = MagicMock()
    mock_fetcher_instance.get.return_value = mock_page

    mock_fetcher_class = MagicMock(return_value=mock_fetcher_instance)

    # Patch the Fetcher at the module where it's imported
    mock_defaults = MagicMock()
    mock_defaults.Fetcher = mock_fetcher_class

    with patch.dict(sys.modules, {"scrapling.defaults": mock_defaults}):
        result = await service.extract("https://example.com/article")

    assert result.title == "Test Article"
    assert result.word_count > 0
    assert result.metadata["domain"] == "example.com"
