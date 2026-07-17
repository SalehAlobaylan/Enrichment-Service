"""Unit tests for RelatedService — dense retrieval orchestration."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.common.config import Settings
from src.retrieval.models.embedder import EmbedderWrapper
from src.retrieval.schemas.related import RelatedRequest
from src.retrieval.services.related import RelatedService


@pytest.fixture
def settings() -> Settings:
    return Settings(
        SERVICE_AUTH_TOKEN="t",
        CMS_SERVICE_TOKEN="c",
        CMS_BASE_URL="http://localhost:8080",
        RELATED_K_DENSE_DEFAULT=50,
    )


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock(spec=EmbedderWrapper)
    embedder.model_name = "Qwen/Qwen3-Embedding-0.6B"
    embedder.dimensions = 1024
    embedder.encode.return_value = {"dense": [[0.1] * 1024], "sparse": None}
    embedder.space_descriptor.return_value = {"space_id": "space-text"}
    return embedder


@pytest.fixture
def mock_cms() -> AsyncMock:
    cms = AsyncMock()
    cms.knn_dense.return_value = []
    return cms


@pytest.fixture
def service(mock_embedder, mock_cms, settings) -> RelatedService:
    return RelatedService(mock_embedder, mock_cms, settings)


def test_dense_results_preserve_dense_order_and_provenance(service: RelatedService) -> None:
    results = service._dense_results(
        [
            {"id": "first", "type": "NEWS", "format": "ARTICLE", "score": 0.9},
            {"id": "second", "type": "NEWS", "score": 0.8},
            {"id": "first", "type": "NEWS", "score": 0.7},
        ]
    )
    assert [item.content_id for item in results] == ["first", "second"]
    assert [item.score for item in results] == [0.9, 0.8]
    assert results[0].content_format == "ARTICLE"
    assert all(item.sources == ["dense"] for item in results)


@pytest.mark.asyncio
async def test_related_text_path_embeds_then_queries_dense_only(
    service: RelatedService, mock_embedder: MagicMock, mock_cms: AsyncMock
) -> None:
    mock_cms.knn_dense.return_value = [{"id": "a", "type": "NEWS", "score": 0.9}]

    response = await service.related(
        RelatedRequest(text="Saudi climate", types=["NEWS"], k=5)
    )

    mock_embedder.encode.assert_called_once_with(["Saudi climate"], False)
    mock_cms.knn_dense.assert_awaited_once()
    assert len(response.results) == 1
    assert response.results[0].content_id == "a"
    assert response.results[0].sources == ["dense"]


@pytest.mark.asyncio
async def test_related_content_id_path_skips_embed(
    service: RelatedService, mock_embedder: MagicMock, mock_cms: AsyncMock
) -> None:
    mock_cms.get_content_embeddings.return_value = {
        "embedding": [0.1] * 1024,
        "embedding_space_id": "space-text",
    }

    await service.related(RelatedRequest(content_id="anchor-1", k=3))

    mock_cms.get_content_embeddings.assert_awaited_once_with("anchor-1")
    mock_embedder.encode.assert_not_called()


@pytest.mark.asyncio
async def test_related_filters_passthrough_to_dense_cms(
    service: RelatedService, mock_cms: AsyncMock
) -> None:
    await service.related(
        RelatedRequest(
            text="x",
            types=["NEWS"],
            formats=["ARTICLE", "TWEET"],
            exclude_ids=["anchor-1", "shown-1"],
            k=3,
        )
    )

    dense_kwargs = mock_cms.knn_dense.call_args.kwargs
    assert dense_kwargs["types"] == ["NEWS"]
    assert dense_kwargs["formats"] == ["ARTICLE", "TWEET"]
    assert dense_kwargs["exclude_ids"] == ["anchor-1", "shown-1"]
    assert dense_kwargs["k"] == 50


@pytest.mark.asyncio
async def test_related_ignores_legacy_sparse_anchor_data(
    service: RelatedService, mock_cms: AsyncMock
) -> None:
    mock_cms.get_content_embeddings.return_value = {
        "embedding": [0.1] * 1024,
        "embedding_space_id": "space-text",
        "embedding_sparse": {"100": 0.5},
    }
    mock_cms.knn_dense.return_value = [{"id": "x", "type": "NEWS", "score": 0.7}]

    response = await service.related(RelatedRequest(content_id="legacy", k=5))

    assert len(response.results) == 1
    assert response.results[0].sources == ["dense"]


@pytest.mark.asyncio
async def test_related_truncates_to_request_k(
    service: RelatedService, mock_cms: AsyncMock
) -> None:
    mock_cms.knn_dense.return_value = [
        {"id": f"d{i}", "type": "NEWS", "score": 1 - i / 100}
        for i in range(20)
    ]

    response = await service.related(RelatedRequest(text="x", k=3))
    assert len(response.results) == 3
