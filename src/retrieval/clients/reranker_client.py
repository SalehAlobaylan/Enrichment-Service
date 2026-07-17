"""HTTP client for a remote reranker deployment.

⚠️  TEMP WORKAROUND (Cranl fixed-instance-RAM limit).
This is a drop-in replacement for the in-process `RerankerWrapper`: it exposes
the SAME surface (`model_name`, `is_loaded`, `load()`, `rerank()`) so
`RelatedService` and the routes don't change. It's used by the "api"-role
instance when `RERANKER_BASE_URL` is set, to call the separate
"enrichment-reranker" deployment over HTTP instead of loading the ~3GB
cross-encoder locally (which would OOM alongside BGE-M3).

`rerank()` is intentionally SYNC (blocking httpx) because the caller invokes it
via `asyncio.to_thread(...)`, exactly like the local wrapper — so this is a
true drop-in. Failures raise; the caller degrades to dense-kNN order.

REMOVE WHEN: a single instance can hold both models (~8GB). Clear
RERANKER_BASE_URL → ModelManager uses the in-process RerankerWrapper again and
this client is dead code.
"""
import httpx

from src.common.utils.logging import get_logger

logger = get_logger(__name__)


class RerankerClient:
    def __init__(
        self,
        base_url: str,
        token: str = "",
        model_name: str = "BAAI/bge-reranker-v2-m3",
        # 30s was too tight for the constrained reranker instance (it timed out
        # before the cross-encoder finished, silently degrading to dense kNN). 50s
        # gives headroom while staying under the ~60s gateway/ingress cap. With
        # RERANK_INPUT_K lowered to ~8, a batch now finishes in ~15s anyway.
        timeout_sec: float = 50.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._model_name = model_name
        self._timeout = timeout_sec
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_sec, connect=min(5.0, timeout_sec)),
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        # "Loaded" == configured. We don't ping on every check; a transient
        # remote failure surfaces as a rerank exception, which RelatedService
        # already catches and degrades to dense-kNN order.
        return bool(self._base_url)

    def load(self) -> None:
        # Remote model — nothing to load locally. Kept for wrapper parity so
        # ModelManager.warmup() can call it uniformly.
        logger.info("reranker_remote_configured", base_url=self._base_url)

    def close(self) -> None:
        """Release the pooled keep-alive transport during role shutdown."""
        self._client.close()

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Score candidates against the query via the remote reranker.

        SYNC on purpose — called inside asyncio.to_thread by RelatedService.
        """
        if not candidates:
            return []

        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        resp = self._client.post(
            f"{self._base_url}/v1/rerank",
            json={"query": query, "candidates": candidates},
            headers=headers,
        )
        resp.raise_for_status()
        return [float(s) for s in resp.json().get("scores", [])]
