from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.common.clients.cms import CMSClient
from src.common.config import Settings
from src.common.workload_admission import WorkloadAdmission
from src.llm.clients.llm import LLMClient
from src.main import app


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        SERVICE_AUTH_TOKEN="test-token",
        ENRICHMENT_RESTART_TOKEN="test-restart-token",
        CMS_SERVICE_TOKEN="test-cms-token",
        CMS_BASE_URL="http://localhost:8080",
        MODELS_DIR="./test-models",
        ENV="test",
    )


@pytest.fixture
def mock_model_manager() -> MagicMock:
    """Mock for Enrichment-Service's ModelManager (text embedder + reranker).

    Whisper + CLIP mocks moved to Media-Service's conftest.
    """
    manager = MagicMock()
    manager.is_ready = {"embedder": True, "reranker": True}
    manager.all_ready = True

    # Embedder mock — Qwen3-Embedding-0.6B (1024-dim, multilingual, dense-only).
    manager.embedder.is_loaded = True
    manager.embedder.model_name = "Qwen/Qwen3-Embedding-0.6B"
    manager.embedder.dimensions = 1024
    manager.embedder.space_descriptor.return_value = {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "test-revision-immutable",
        "dimensions": 1024,
        "space_id": "qwen-test-space-v1",
        "producer_id": "qwen-test-producer-v1",
    }

    # Reranker mock — bge-reranker-v2-m3 (Slice B). Default scores are
    # decreasing, so when the orchestration test re-ranks N candidates the
    # output order matches input. Individual tests override .rerank as needed.
    manager.reranker.is_loaded = True
    manager.reranker.model_name = "BAAI/bge-reranker-v2-m3"
    manager.reranker.rerank = MagicMock(return_value=[])

    return manager


@pytest.fixture
def mock_cms_client() -> AsyncMock:
    client = AsyncMock(spec=CMSClient)
    client.health_check.return_value = True
    client.store_embedding.return_value = {"ok": True}
    client.update_content.return_value = {"ok": True}
    client.merge_enrichment_metadata.return_value = {"success": True}
    # Slice A/B defaults — individual tests override as needed.
    client.get_content_embeddings.return_value = {
        "embedding": [0.1] * 1024,
        "embedding_space_id": "qwen-test-space-v1",
    }
    client.knn_dense.return_value = []
    client.batch_text.return_value = []
    client.get_content_item_basic.return_value = {
        "id": "anchor-id",
        "type": "ARTICLE",
        "title": "Test anchor",
        "excerpt": "Test excerpt",
        "source_name": "test-source",
        "published_at": "2026-05-20T12:00:00Z",
    }
    return client


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    return AsyncMock(spec=LLMClient)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def client(
    test_settings: Settings,
    mock_model_manager: MagicMock,
    mock_cms_client: AsyncMock,
    mock_llm_client: AsyncMock,
) -> TestClient:
    app.state.settings = test_settings
    app.state.model_manager = mock_model_manager
    app.state.cms_client = mock_cms_client
    app.state.llm_client = mock_llm_client
    app.state.workload_admission = WorkloadAdmission()
    return TestClient(app, raise_server_exceptions=False)
