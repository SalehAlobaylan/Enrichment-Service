import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.services.tagging import TagsResult
from src.retrieval.models.embedder import MAX_TEXT_LENGTH, EmbedderWrapper
from src.retrieval.services.embedding import EmbeddingService


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock(spec=EmbedderWrapper)
    embedder.model_name = "Qwen/Qwen3-Embedding-0.6B"
    embedder.dimensions = 1024
    # BGE-M3 returns dict shape: {dense: [[1024]], sparse: [{token: weight}] | None}.
    # Default fixture mimics the with_sparse=True path.
    embedder.encode.return_value = {
        "dense": [[0.1] * 1024, [0.2] * 1024],
        "sparse": [{"100": 0.5, "200": 0.3}, {"300": 0.4}],
    }
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
    assert result.dimensions == 1024
    assert result.model == "Qwen/Qwen3-Embedding-0.6B"


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
async def test_optional_tags_do_not_block_required_embedding_writeback(
    mock_embedder: MagicMock, mock_cms: AsyncMock
) -> None:
    """Tag extraction is eventual; the required embedding receipt is not."""
    mock_embedder.encode.return_value = {
        "dense": [[0.1] * 1024],
        "sparse": None,
    }
    tagger = MagicMock()
    tagger.extract = AsyncMock(return_value=TagsResult(tags=["technology"]))
    service = EmbeddingService(mock_embedder, mock_cms, tagger=tagger)

    result = await service.embed(
        ["a sufficiently long source text"],
        content_ids=["id-1"],
        extract_tags=True,
    )

    assert result.write_back_status == "ok"
    mock_cms.store_embedding.assert_awaited_once()
    assert mock_cms.store_embedding.call_args.kwargs["topic_tags"] is None

    # Let the detached optional task finish after the required writeback has
    # already returned. The test intentionally checks the two operations are
    # separate so a stage-correlated embedding cannot be written twice.
    for _ in range(10):
        if mock_cms.update_topic_tags.await_count:
            break
        await asyncio.sleep(0)
    mock_cms.update_topic_tags.assert_awaited_once_with("id-1", ["technology"])


@pytest.mark.asyncio
async def test_optional_tags_are_cancelled_when_required_writeback_fails(
    mock_embedder: MagicMock, mock_cms: AsyncMock
) -> None:
    """A failed required receipt must not leave an orphan LLM task running."""
    mock_embedder.encode.return_value = {
        "dense": [[0.1] * 1024],
        "sparse": None,
    }
    mock_cms.store_embedding.side_effect = RuntimeError("CMS unavailable")
    cancelled = asyncio.Event()

    async def blocked_tagging(_: str) -> TagsResult:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    tagger = MagicMock()
    tagger.extract = blocked_tagging
    service = EmbeddingService(mock_embedder, mock_cms, tagger=tagger)

    result = await service.embed(
        ["a sufficiently long source text"],
        content_ids=["id-1"],
        extract_tags=True,
    )

    assert result.write_back_status == "failed"
    await asyncio.wait_for(cancelled.wait(), timeout=0.5)
    mock_cms.update_topic_tags.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_query(service: EmbeddingService, mock_embedder: MagicMock) -> None:
    # embed_query calls encode(texts, False) — no sparse needed for query path.
    mock_embedder.encode.return_value = {"dense": [[0.5] * 1024], "sparse": None}
    result = await service.embed_query("search query")
    assert len(result.embedding) == 1024


@pytest.mark.asyncio
async def test_embed_sparse_passed_to_writeback(
    service: EmbeddingService, mock_embedder: MagicMock, mock_cms: AsyncMock
) -> None:
    """When extract_sparse=True, the sparse map reaches cms.store_embedding."""
    mock_embedder.encode.return_value = {
        "dense": [[0.1] * 1024],
        "sparse": [{"100": 0.5}],
    }
    await service.embed(
        ["hello"],
        content_ids=["id-1"],
        extract_sparse=True,
    )
    mock_cms.store_embedding.assert_called_once()
    call_kwargs = mock_cms.store_embedding.call_args.kwargs
    assert call_kwargs["embedding_sparse"] is not None
    assert isinstance(call_kwargs["embedding_sparse"], dict)


@pytest.mark.asyncio
async def test_embed_sparse_none_when_not_requested(
    service: EmbeddingService, mock_embedder: MagicMock, mock_cms: AsyncMock
) -> None:
    """extract_sparse=False (default) means encode() returns sparse=None and
    the CMS write-back gets embedding_sparse=None — the column stays NULL."""
    mock_embedder.encode.return_value = {"dense": [[0.1] * 1024], "sparse": None}
    await service.embed(["hello"], content_ids=["id-1"])
    call_kwargs = mock_cms.store_embedding.call_args.kwargs
    assert call_kwargs["embedding_sparse"] is None


@pytest.mark.asyncio
async def test_writeback_rejects_mismatched_vectors_without_cms_call(
    service: EmbeddingService, mock_cms: AsyncMock
) -> None:
    status, error = await service._write_back(
        ["id-1", "id-2"], [[0.1] * 1024], None
    )
    assert (status, error) == ("failed", "writeback_length_mismatch")
    mock_cms.store_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_writeback_reports_partial_failure_without_dropping_other_items(
    service: EmbeddingService, mock_cms: AsyncMock
) -> None:
    mock_cms.store_embedding.side_effect = [None, RuntimeError("CMS unavailable")]
    status, error = await service._write_back(
        ["id-1", "id-2"], [[0.1] * 1024, [0.2] * 1024], None
    )
    assert (status, error) == ("failed", "cms_writeback_failed")
    assert mock_cms.store_embedding.await_count == 2


def test_text_truncation() -> None:
    long_text = "a" * (MAX_TEXT_LENGTH + 1000)
    assert len(long_text[:MAX_TEXT_LENGTH]) == MAX_TEXT_LENGTH
