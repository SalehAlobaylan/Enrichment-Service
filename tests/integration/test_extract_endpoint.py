from unittest.mock import patch

from fastapi.testclient import TestClient

from src.extraction.schemas.extract import ExtractResponse


def test_extract_success(client: TestClient, auth_headers: dict[str, str]) -> None:
    mock_result = ExtractResponse(
        title="Test",
        text="Article body",
        author="Author",
        published_at=None,
        excerpt="Summary",
        site_name="example.com",
        image_url=None,
        word_count=2,
        html=None,
        metadata={"url": "https://example.com", "domain": "example.com"},
    )
    with patch(
        "src.extraction.services.extraction.ExtractionService.extract",
        return_value=mock_result,
    ):
        resp = client.post(
            "/v1/extract",
            json={"url": "https://example.com/article"},
            headers=auth_headers,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Test"
    assert data["text"] == "Article body"
    assert data["word_count"] == 2


def test_extract_requires_url(client: TestClient, auth_headers: dict[str, str]) -> None:
    resp = client.post(
        "/v1/extract",
        json={"url": ""},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_extract_requires_auth(client: TestClient) -> None:
    resp = client.post("/v1/extract", json={"url": "https://example.com"})
    assert resp.status_code in (401, 403)
