from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.embedder import MAX_TEXT_LENGTH, EmbedderWrapper
from src.services.embedding import EmbeddingService


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock(spec=EmbedderWrapper)
    embedder.model_name = "all-MiniLM-L6-v2"
    embedder.dimensions = 384
    embedder.encode.return_value = [[0.1] * 384, [0.2] * 384]
    return embedder


@pytest.fixture
def mock_cms() -> AsyncMock:
    cms = AsyncMock()
    cms.store_embedding.return_value = {"ok": True}
    return cms


@pytest.fixture
def service(mock_embedder: MagicMock, mock_cms: AsyncMock) -> EmbeddingService:
    return EmbeddingService(mock_embedder, mock_cms)


@pytest.mark.asyncio
async def test_embed_returns_vectors(service: EmbeddingService) -> None:
    result = await service.embed(["hello", "world"])
    assert len(result.embeddings) == 2
    assert result.dimensions == 384
    assert result.model == "all-MiniLM-L6-v2"


@pytest.mark.asyncio
async def test_embed_no_writeback_without_content_ids(
    service: EmbeddingService, mock_cms: AsyncMock
) -> None:
    await service.embed(["hello"])
    mock_cms.store_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_embed_writeback_with_content_ids(
    service: EmbeddingService, mock_cms: AsyncMock
) -> None:
    await service.embed(["hello", "world"], content_ids=["id-1", "id-2"])
    assert mock_cms.store_embedding.call_count == 2


@pytest.mark.asyncio
async def test_embed_query(service: EmbeddingService, mock_embedder: MagicMock) -> None:
    mock_embedder.encode.return_value = [[0.5] * 384]
    result = await service.embed_query("search query")
    assert len(result.embedding) == 384


def test_text_truncation() -> None:
    long_text = "a" * (MAX_TEXT_LENGTH + 1000)
    assert len(long_text[:MAX_TEXT_LENGTH]) == MAX_TEXT_LENGTH
