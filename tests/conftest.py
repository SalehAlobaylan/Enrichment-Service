from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.common.clients.cms import CMSClient
from src.common.config import Settings
from src.llm.clients.llm import LLMClient
from src.main import app


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        SERVICE_AUTH_TOKEN="test-token",
        CMS_SERVICE_TOKEN="test-cms-token",
        CMS_BASE_URL="http://localhost:8080",
        MODELS_DIR="./test-models",
        ENV="test",
    )


@pytest.fixture
def mock_model_manager() -> MagicMock:
    """Mock for Enrichment-Service's ModelManager (text embedder only).

    Whisper + CLIP mocks moved to Media-Service's conftest.
    """
    manager = MagicMock()
    manager.is_ready = {"embedder": True}
    manager.all_ready = True

    # Embedder mock — BGE-M3 (1024-dim, multilingual) after Slice 0.
    manager.embedder.is_loaded = True
    manager.embedder.model_name = "BAAI/bge-m3"
    manager.embedder.dimensions = 1024

    return manager


@pytest.fixture
def mock_cms_client() -> AsyncMock:
    client = AsyncMock(spec=CMSClient)
    client.health_check.return_value = True
    client.store_embedding.return_value = {"ok": True}
    client.update_content.return_value = {"ok": True}
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
    return TestClient(app, raise_server_exceptions=False)
