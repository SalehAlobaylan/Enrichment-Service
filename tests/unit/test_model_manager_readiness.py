from unittest.mock import MagicMock

from src.common.config import Settings
from src.retrieval.clients.reranker_client import RerankerClient
from src.retrieval.models.manager import ModelManager
from src.retrieval.models.reranker import RerankerWrapper


def _manager_with_embedder(descriptor: dict, *, dimensions: int = 1024) -> ModelManager:
    manager = ModelManager(
        Settings(ENRICHMENT_ROLE="api", RERANKER_BASE_URL="", LLM_FALLBACK_PROVIDERS="")
    )
    manager.embedder = MagicMock()
    manager.embedder.is_loaded = True
    manager.embedder.dimensions = dimensions
    manager.embedder.space_descriptor.return_value = descriptor
    return manager


def test_api_readiness_requires_stable_embedding_lifecycle_identity() -> None:
    unresolved = _manager_with_embedder(
        {"revision": "", "space_id": "", "producer_id": ""}
    )
    assert unresolved.all_ready is False

    ready = _manager_with_embedder(
        {"revision": "immutable-revision", "space_id": "space", "producer_id": "producer"}
    )
    assert ready.all_ready is True


def test_api_readiness_rejects_wrong_embedding_dimension() -> None:
    manager = _manager_with_embedder(
        {"revision": "immutable-revision", "space_id": "space", "producer_id": "producer"},
        dimensions=512,
    )
    assert manager.all_ready is False


def test_reranker_role_does_not_construct_an_embedder() -> None:
    manager = ModelManager(Settings(ENRICHMENT_ROLE="reranker", LLM_FALLBACK_PROVIDERS=""))
    assert manager.embedder is None
    assert set(manager.is_ready) == {"reranker"}


def test_api_local_and_remote_reranker_models_are_selected_from_one_settings_object() -> None:
    local = ModelManager(Settings(ENRICHMENT_ROLE="api", RERANKER_BASE_URL=""))
    assert local.embedder is not None
    assert isinstance(local.reranker, RerankerWrapper)

    remote = ModelManager(
        Settings(
            ENRICHMENT_ROLE="api",
            RERANKER_BASE_URL="https://reranker.internal",
            RERANKER_SERVICE_TOKEN="relationship-token",
        )
    )
    assert remote.embedder is not None
    assert isinstance(remote.reranker, RerankerClient)
    remote.close()
