from typing import Any, cast

import httpx

from src.common.clients.circuit_breaker import CircuitBreaker
from src.common.config import Settings
from src.common.middleware.request_id import current_request_id
from src.common.utils.logging import get_logger
from src.common.utils.metrics import cms_writeback_total

logger = get_logger(__name__)


def _is_countable_cms_failure(exc: Exception) -> bool:
    """Only dependency availability failures may open the shared CMS breaker."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = cast(httpx.HTTPStatusError, exc).response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.HTTPError)


class CMSClient:
    def __init__(self, settings: Settings):
        raw_base_url = settings.CMS_BASE_URL.rstrip("/")
        self.base_url = raw_base_url
        self.public_base_url = (
            raw_base_url.removesuffix("/internal")
            if raw_base_url.endswith("/internal")
            else raw_base_url
        )
        # This client is the Enrichment principal at CMS. The legacy token is
        # accepted only by Settings' development/rollout compatibility path.
        self.token = settings.cms_writeback_token
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=settings.CB_FAILURE_THRESHOLD,
            reset_timeout_sec=settings.CB_RESET_TIMEOUT_SEC,
            half_open_requests=settings.CB_HALF_OPEN_REQUESTS,
            metric_name="cms_core",
        )
        # AI-spend delivery is best-effort. Its own dependency failures must
        # never open the core breaker that protects retrieval and write-back.
        self.telemetry_circuit_breaker = CircuitBreaker(
            failure_threshold=settings.CB_FAILURE_THRESHOLD,
            reset_timeout_sec=settings.CB_RESET_TIMEOUT_SEC,
            half_open_requests=settings.CB_HALF_OPEN_REQUESTS,
            metric_name="cms_telemetry",
        )
        # Optional tag persistence must not open the core breaker used by
        # required embeddings, retrieval, and lifecycle write-backs.
        self.optional_metadata_circuit_breaker = CircuitBreaker(
            failure_threshold=settings.CB_FAILURE_THRESHOLD,
            reset_timeout_sec=settings.CB_RESET_TIMEOUT_SEC,
            half_open_requests=settings.CB_HALF_OPEN_REQUESTS,
            metric_name="cms_optional_metadata",
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
            # CMS /health is aggregate operational readiness and includes this
            # service. Use its dependency-independent liveness probe here so a
            # temporary readiness fault cannot form a circular outage.
            resp = await self.client.get(f"{self.public_base_url}/live")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def store_embedding(
        self,
        content_id: str,
        embedding: list[float],
        topic_tags: list[str] | None = None,
        embedding_sparse: dict[str, float] | None = None,
        model: str | None = None,
        space_id: str | None = None,
        producer_id: str | None = None,
        artifact_recovery: dict[str, str] | None = None,
        content_stage: dict[str, str] | None = None,
        pipeline_repair: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Write text embedding to CMS.

        embedding         — 1024-dim dense vector
        embedding_sparse  — legacy BGE-M3 lexical weights; Qwen is dense-only,
                            so this is always omitted now
        topic_tags        — optional topic tags extracted by tagging service
        model             — embedder name (provenance display label).
        space_id/producer_id — immutable vector-space identities (stage 10). Sent
                            only when resolved (non-empty); an unresolved space
                            leaves the row unstamped debt rather than stamping a
                            false-stable identity.
        """
        payload: dict[str, Any] = {
            "embedding": embedding,
            "topic_tags": topic_tags or [],
        }
        if embedding_sparse:
            payload["embedding_sparse"] = embedding_sparse
        if model:
            payload["model"] = model
        if space_id:
            payload["space_id"] = space_id
        if producer_id:
            payload["producer_id"] = producer_id
        if artifact_recovery:
            payload["artifact_recovery"] = artifact_recovery
        if pipeline_repair:
            payload["pipeline_repair"] = pipeline_repair
        if content_stage:
            payload["content_stage"] = content_stage
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

    async def update_topic_tags(self, content_id: str, topic_tags: list[str]) -> dict[str, Any]:
        """Persist optional tags after the required embedding writeback.

        This endpoint intentionally has no content-stage correlation. The
        embedding write owns the required stage receipt; tags are an
        idempotent optional enrichment effect and must not extend that lease.
        """
        return await self._request(
            "PATCH",
            f"/internal/content-items/{content_id}/topic-tags",
            json={"topic_tags": topic_tags},
            metric_label="update_topic_tags",
            circuit_breaker=self.optional_metadata_circuit_breaker,
        )

    async def merge_enrichment_metadata(
        self,
        content_id: str,
        fields: dict[str, Any],
        artifact_recovery: dict[str, str] | None = None,
        content_stage: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Persist only the fields Enrichment owns, preserving all other metadata."""
        return await self._request(
            "PATCH",
            f"/internal/content-items/{content_id}/enrichment-metadata",
            json={
                "fields": fields,
                **({"artifact_recovery": artifact_recovery} if artifact_recovery else {}),
                **({"content_stage": content_stage} if content_stage else {}),
            },
            metric_label="merge_enrichment_metadata",
        )

    async def emit_ai_spend_events(self, events: list[dict[str, Any]]) -> None:
        """Best-effort governor metering. Callers schedule this and never await it.

        A failed ledger write must not turn a successful enrichment operation
        into a user-visible failure.
        """
        await self._request(
            "POST",
            "/internal/ai-spend/events",
            json={"events": events},
            metric_label="emit_ai_spend_events",
            circuit_breaker=self.telemetry_circuit_breaker,
        )

    async def put_pipeline_lane_snapshot(
        self, lane: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Publish bounded Enrichment lane telemetry; CMS remains the owner."""
        return await self._request(
            "PUT",
            f"/internal/pipeline-lanes/{lane}/snapshot",
            json={"lane": lane, **payload},
            metric_label="put_pipeline_lane_snapshot",
        )

    # ─── Slice A: dense retrieval ───────────────────────────────────

    async def get_content_embeddings(self, content_id: str) -> dict[str, Any]:
        """GET /internal/content-items/:id/embeddings — fetch dense vector identity.

        Used by /v1/related when the caller passes content_id instead of text
        — avoids re-embedding what's already stored. Returns:
            {"embedding": list[float] | None,
             "embedding_space_id": str | None}
        """
        return await self._request(
            "GET",
            f"/internal/content-items/{content_id}/embeddings",
            metric_label="get_content_embeddings",
        )

    async def knn_dense(
        self,
        vector: list[float],
        space_id: str,
        types: list[str] | None = None,
        formats: list[str] | None = None,
        k: int = 50,
        exclude_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """POST /internal/content-items/knn — cosine kNN against `embedding`.

        Returns the `hits` list directly (unwrapped from CMS's response
        envelope). Each hit is `{id, type, score}` where score is `1 - cosine_distance`.
        """
        payload: dict[str, Any] = {
            "embedding": vector,
            "space_id": space_id,
            "types": types or [],
            "formats": formats or [],
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

    # ─── Slice B: reranker batch text + anchor fetch ────────────────

    async def batch_text(self, ids: list[str]) -> list[dict[str, Any]]:
        """POST /internal/content-items/batch-text — fetch text + metadata
        for a small set of ids (typically the post-dense-kNN candidate pool that
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
        circuit_breaker: CircuitBreaker | None = None,
    ) -> dict[str, Any]:
        async def _do_request() -> dict[str, Any]:
            url = self._build_url(path)
            request_id = current_request_id()
            headers = {"X-Request-ID": request_id} if request_id else None
            resp = await self.client.request(method, url, json=json, headers=headers)
            resp.raise_for_status()
            return resp.json()

        try:
            result = await (circuit_breaker or self.circuit_breaker).execute(
                _do_request, count_failure=_is_countable_cms_failure
            )
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
