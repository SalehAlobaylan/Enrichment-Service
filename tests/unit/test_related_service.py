"""Unit tests for RelatedService — RRF fusion correctness + orchestration."""
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
        RRF_K=60,
        RELATED_K_DENSE_DEFAULT=50,
        RELATED_K_SPARSE_DEFAULT=50,
    )


@pytest.fixture
def mock_embedder() -> MagicMock:
    e = MagicMock(spec=EmbedderWrapper)
    e.model_name = "BAAI/bge-m3"
    e.dimensions = 1024
    e.encode.return_value = {
        "dense": [[0.1] * 1024],
        "sparse": [{"100": 0.5, "200": 0.3}],
    }
    return e


@pytest.fixture
def mock_cms() -> AsyncMock:
    cms = AsyncMock()
    cms.knn_dense.return_value = []
    cms.knn_sparse.return_value = []
    return cms


@pytest.fixture
def service(mock_embedder, mock_cms, settings) -> RelatedService:
    return RelatedService(mock_embedder, mock_cms, settings)


# ─── RRF fusion correctness ────────────────────────────────


def test_rrf_fusion_combines_rankings_in_order(service: RelatedService) -> None:
    """Item appearing in both rankings beats item appearing in only one."""
    dense = [
        {"id": "a", "type": "TWEET", "score": 0.9},
        {"id": "b", "type": "TWEET", "score": 0.8},
        {"id": "c", "type": "COMMENT", "score": 0.7},
    ]
    sparse = [
        {"id": "b", "type": "TWEET", "score": 0.95},
        {"id": "d", "type": "COMMENT", "score": 0.85},
        {"id": "a", "type": "TWEET", "score": 0.6},
    ]
    fused = service._rrf_fuse(dense, sparse, k=60)

    ids = [r.content_id for r in fused]
    # "b" in both at high ranks → wins; "a" in both → second; "c" / "d" tail
    assert ids[0] == "b"
    assert ids[1] == "a"
    assert set(ids[2:]) == {"c", "d"}


def test_rrf_marks_sources_correctly(service: RelatedService) -> None:
    dense = [
        {"id": "only_dense", "type": "TWEET", "score": 0.9},
        {"id": "both", "type": "TWEET", "score": 0.8},
    ]
    sparse = [
        {"id": "both", "type": "TWEET", "score": 0.95},
        {"id": "only_sparse", "type": "COMMENT", "score": 0.85},
    ]
    fused = service._rrf_fuse(dense, sparse, k=60)
    by_id = {r.content_id: r for r in fused}
    assert by_id["both"].sources == ["dense", "sparse"]
    assert by_id["only_dense"].sources == ["dense"]
    assert by_id["only_sparse"].sources == ["sparse"]


def test_rrf_empty_inputs_returns_empty(service: RelatedService) -> None:
    assert service._rrf_fuse([], [], k=60) == []


def test_rrf_score_formula(service: RelatedService) -> None:
    """Item at rank 0 in one ranking only should score ~1/(60+1).

    _rrf_fuse rounds to 6 decimal places for response compactness, so the
    comparison tolerance must be ≥1e-6.
    """
    dense = [{"id": "x", "type": "TWEET"}]
    fused = service._rrf_fuse(dense, [], k=60)
    assert fused[0].score == pytest.approx(1.0 / 61, abs=1e-6)


# ─── Orchestration ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_related_text_path_embeds_then_fans_out(
    service: RelatedService,
    mock_embedder: MagicMock,
    mock_cms: AsyncMock,
) -> None:
    mock_cms.knn_dense.return_value = [{"id": "a", "type": "TWEET"}]
    mock_cms.knn_sparse.return_value = [{"id": "a", "type": "TWEET"}]

    resp = await service.related(
        RelatedRequest(text="Saudi climate", types=["TWEET"], k=5)
    )

    mock_embedder.encode.assert_called_once()
    # Both knn paths exercised in parallel
    mock_cms.knn_dense.assert_awaited_once()
    mock_cms.knn_sparse.assert_awaited_once()
    assert len(resp.results) == 1
    assert resp.results[0].content_id == "a"


@pytest.mark.asyncio
async def test_related_content_id_path_skips_embed(
    service: RelatedService,
    mock_embedder: MagicMock,
    mock_cms: AsyncMock,
) -> None:
    mock_cms.get_content_embeddings.return_value = {
        "embedding": [0.1] * 1024,
        "embedding_sparse": {"100": 0.5},
    }
    mock_cms.knn_dense.return_value = []
    mock_cms.knn_sparse.return_value = []

    await service.related(RelatedRequest(content_id="anchor-1", k=3))

    # Anchor fetch happened — local embedder NOT called
    mock_cms.get_content_embeddings.assert_awaited_once_with("anchor-1")
    mock_embedder.encode.assert_not_called()


@pytest.mark.asyncio
async def test_related_filters_passthrough_to_cms(
    service: RelatedService,
    mock_cms: AsyncMock,
) -> None:
    mock_cms.knn_dense.return_value = []
    mock_cms.knn_sparse.return_value = []

    await service.related(
        RelatedRequest(
            text="x",
            types=["TWEET", "COMMENT"],
            exclude_ids=["anchor-1", "shown-1"],
            k=3,
        )
    )

    dense_kwargs = mock_cms.knn_dense.call_args.kwargs
    assert dense_kwargs["types"] == ["TWEET", "COMMENT"]
    assert dense_kwargs["exclude_ids"] == ["anchor-1", "shown-1"]
    # Per-mode k comes from settings.RELATED_K_*_DEFAULT, NOT request.k
    assert dense_kwargs["k"] == 50


@pytest.mark.asyncio
async def test_related_anchor_without_sparse_degrades_gracefully(
    service: RelatedService,
    mock_cms: AsyncMock,
) -> None:
    """If an anchor has dense but no sparse (backfill not done yet), the
    sparse kNN gets called with an empty map and returns nothing — RRF
    falls through to dense-only."""
    mock_cms.get_content_embeddings.return_value = {
        "embedding": [0.1] * 1024,
        "embedding_sparse": None,
    }
    mock_cms.knn_dense.return_value = [{"id": "x", "type": "TWEET"}]
    mock_cms.knn_sparse.return_value = []

    resp = await service.related(RelatedRequest(content_id="legacy", k=5))
    assert len(resp.results) == 1
    assert resp.results[0].sources == ["dense"]


@pytest.mark.asyncio
async def test_related_truncates_to_request_k(
    service: RelatedService, mock_cms: AsyncMock
) -> None:
    mock_cms.knn_dense.return_value = [
        {"id": f"d{i}", "type": "TWEET"} for i in range(20)
    ]
    mock_cms.knn_sparse.return_value = []

    resp = await service.related(RelatedRequest(text="x", k=3))
    assert len(resp.results) == 3
