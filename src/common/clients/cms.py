from typing import Any

import httpx

from src.common.clients.circuit_breaker import CircuitBreaker
from src.common.config import Settings
from src.common.middleware.request_id import current_request_id
from src.common.utils.logging import get_logger
from src.common.utils.metrics import cms_writeback_total

logger = get_logger(__name__)


class CMSClient:
    def __init__(self, settings: Settings):
        raw_base_url = settings.CMS_BASE_URL.rstrip("/")
        self.base_url = raw_base_url
        self.public_base_url = (
            raw_base_url.removesuffix("/internal")
            if raw_base_url.endswith("/internal")
            else raw_base_url
        )
        self.token = settings.CMS_SERVICE_TOKEN
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=settings.CB_FAILURE_THRESHOLD,
            reset_timeout_sec=settings.CB_RESET_TIMEOUT_SEC,
            half_open_requests=settings.CB_HALF_OPEN_REQUESTS,
        )
        headers = {
            "Content-Type": "application/json",
            "X-Service-Name": "enrichment-service",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self.client = httpx.AsyncClient(
            timeout=settings.CMS_REQUEST_TIMEOUT_SEC,
            headers=headers,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get(f"{self.public_base_url}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def create_transcript(
        self,
        content_item_id: str,
        full_text: str,
        language: str,
        word_timestamps: list[dict] | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "content_item_id": content_item_id,
            "full_text": full_text,
            "language": language,
        }
        if word_timestamps:
            payload["word_timestamps"] = word_timestamps
        if summary:
            payload["summary"] = summary

        return await self._request(
            "POST",
            "/internal/transcripts",
            json=payload,
            metric_label="create_transcript",
        )

    async def link_transcript(self, content_id: str, transcript_id: str) -> dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/internal/content-items/{content_id}/transcript",
            json={"transcript_id": transcript_id},
            metric_label="link_transcript",
        )

    async def store_embedding(
        self,
        content_id: str,
        embedding: list[float],
        topic_tags: list[str] | None = None,
        embedding_sparse: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Write BGE-M3 text embedding to CMS.

        embedding         — 1024-dim dense vector (required after Slice 0)
        embedding_sparse  — BGE-M3 lexical weights {token_id_str: weight}.
                            Forward-compat — Slice A will start populating it
                            once FlagEmbedding lands. Today always omitted.
        topic_tags        — optional topic tags extracted by tagging service
        """
        payload: dict[str, Any] = {
            "embedding": embedding,
            "topic_tags": topic_tags or [],
        }
        if embedding_sparse:
            payload["embedding_sparse"] = embedding_sparse
        return await self._request(
            "PATCH",
            f"/internal/content-items/{content_id}/embedding",
            json=payload,
            metric_label="store_embedding",
        )

    async def update_content(self, content_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        return await self._request(
            "PUT",
            f"/internal/content-items/{content_id}",
            json={"metadata": metadata},
            metric_label="update_content",
        )

    # ─── Slice A: hybrid retrieval ──────────────────────────────────

    async def get_content_embeddings(self, content_id: str) -> dict[str, Any]:
        """GET /internal/content-items/:id/embeddings — fetch (dense, sparse).

        Used by /v1/related when the caller passes content_id instead of text
        — avoids re-embedding what's already stored. Returns:
            {"embedding": list[float] | None,
             "embedding_sparse": dict[str, float] | None}
        """
        return await self._request(
            "GET",
            f"/internal/content-items/{content_id}/embeddings",
            metric_label="get_content_embeddings",
        )

    async def knn_dense(
        self,
        vector: list[float],
        types: list[str] | None = None,
        k: int = 50,
        exclude_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """POST /internal/content-items/knn — cosine kNN against `embedding`.

        Returns the `hits` list directly (unwrapped from CMS's response
        envelope). Each hit is `{id, type, score}` where score is `1 - cosine_distance`.
        """
        payload: dict[str, Any] = {
            "embedding": vector,
            "types": types or [],
            "k": k,
            "exclude_ids": exclude_ids or [],
        }
        resp = await self._request(
            "POST",
            "/internal/content-items/knn",
            json=payload,
            metric_label="knn_dense",
        )
        return resp.get("hits", [])

    async def knn_sparse(
        self,
        sparse_map: dict[str, float],
        types: list[str] | None = None,
        k: int = 50,
        exclude_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """POST /internal/content-items/knn-sparse — inner-product kNN against `embedding_sparse`.

        sparse_map is BGE-M3's lexical-weights output: {token_id_str: weight}.
        Returns the `hits` list directly.
        """
        payload: dict[str, Any] = {
            "embedding_sparse": sparse_map,
            "types": types or [],
            "k": k,
            "exclude_ids": exclude_ids or [],
        }
        resp = await self._request(
            "POST",
            "/internal/content-items/knn-sparse",
            json=payload,
            metric_label="knn_sparse",
        )
        return resp.get("hits", [])

    # ─── Slice B: reranker batch text + anchor fetch ────────────────

    async def batch_text(self, ids: list[str]) -> list[dict[str, Any]]:
        """POST /internal/content-items/batch-text — fetch text + metadata
        for a small set of ids (typically the post-RRF candidate pool that
        the reranker stage scores).

        Returns the `items` list directly. Each item has:
            {id, type, title, excerpt, body_text, source_name, published_at}
        with nullable string fields preserved as None when absent.
        """
        if not ids:
            return []
        resp = await self._request(
            "POST",
            "/internal/content-items/batch-text",
            json={"ids": ids},
            metric_label="batch_text",
        )
        return resp.get("items", [])

    async def get_content_item_basic(self, content_id: str) -> dict[str, Any]:
        """GET /internal/content-items/:id — fetch the full content item.

        Used by FeedNewsService to resolve the anchor for the slide response.
        Returns the raw CMS payload; the caller picks out the fields they
        need (typically just title, excerpt, type, source_name, published_at).
        """
        return await self._request(
            "GET",
            f"/internal/content-items/{content_id}",
            metric_label="get_content_item_basic",
        )

    async def update_status(
        self,
        content_id: str,
        status: str,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": status}
        if failure_reason:
            payload["failure_reason"] = failure_reason
        return await self._request(
            "PATCH",
            f"/internal/content-items/{content_id}/status",
            json=payload,
            metric_label="update_status",
        )

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        metric_label: str = "unknown",
    ) -> dict[str, Any]:
        async def _do_request() -> dict[str, Any]:
            url = self._build_url(path)
            request_id = current_request_id()
            headers = {"X-Request-ID": request_id} if request_id else None
            resp = await self.client.request(method, url, json=json, headers=headers)
            resp.raise_for_status()
            return resp.json()

        try:
            result = await self.circuit_breaker.execute(_do_request)
            cms_writeback_total.labels(endpoint=metric_label, status="success").inc()
            return result
        except Exception as exc:
            cms_writeback_total.labels(endpoint=metric_label, status="failure").inc()
            logger.error(
                "cms_request_failed",
                method=method,
                path=path,
                error=str(exc),
            )
            raise

    def _build_url(self, path: str) -> str:
        if self.base_url.endswith("/internal") and path.startswith("/internal/"):
            return f"{self.base_url}{path.removeprefix('/internal')}"
        return f"{self.base_url}{path}"
