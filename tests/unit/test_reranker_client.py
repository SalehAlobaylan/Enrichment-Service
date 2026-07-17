from unittest.mock import MagicMock, patch

from src.retrieval.clients.reranker_client import RerankerClient


def test_remote_reranker_uses_one_client_with_explicit_bearer_token() -> None:
    client = MagicMock()
    client.post.return_value = MagicMock(
        json=lambda: {"scores": [0.8]},
        raise_for_status=lambda: None,
    )
    with patch("src.retrieval.clients.reranker_client.httpx.Client", return_value=client):
        reranker = RerankerClient("https://reranker.internal", token="relationship-token")
        assert reranker.rerank("query", ["candidate"]) == [0.8]
        reranker.close()

    assert client.post.call_args.kwargs["headers"]["Authorization"] == "Bearer relationship-token"
    client.close.assert_called_once()
