"""Hybrid retrieval orchestration for POST /v1/related.

Pipeline:
  1. Resolve query vector — either fetch (dense, sparse) for `content_id`
     from CMS, or embed `text` via BGE-M3 (one forward pass returns both).
  2. Fan out two parallel kNN queries against CMS (dense vs sparse), using
     candidate sizes from RELATED_K_*_DEFAULT.
  3. Fuse the two rankings with Reciprocal Rank Fusion (RRF) — no weights,
     no calibration, robust across query types and score scales.
  4. Truncate to the caller's `k` and return with per-item provenance.

Why RRF: the dense column uses cosine distance (smaller is better, in
[0, 2]) and the sparse column uses negative inner product (also smaller
better, but a different scale entirely). RRF only looks at RANK, not raw
scores, so cross-mode score normalization is unnecessary. k=60 is the
standard literature value (Cormack, Clarke, Büttcher 2009).
"""
import asyncio
from collections import defaultdict

from src.common.clients.cms import CMSClient
from src.common.config import Settings
from src.common.utils.logging import get_logger
from src.common.utils.metrics import (
    related_duration,
    related_requests_total,
    rrf_fusion_overlap_ratio,
)
from src.retrieval.models.embedder import EmbedderWrapper
from src.retrieval.schemas.related import RelatedItem, RelatedRequest, RelatedResponse

logger = get_logger(__name__)


class RelatedService:
    def __init__(
        self,
        embedder: EmbedderWrapper,
        cms_client: CMSClient,
        settings: Settings,
    ) -> None:
        self.embedder = embedder
        self.cms_client = cms_client
        self.settings = settings

    async def related(self, req: RelatedRequest) -> RelatedResponse:
        with related_duration.time():
            dense_q, sparse_q = await self._resolve_query_vectors(req)

            # Dense + sparse fan-out in parallel. Both calls go through the
            # same CMS circuit breaker, so a CMS outage fails both fast
            # rather than spending 2× the budget on retries.
            dense_task = self.cms_client.knn_dense(
                dense_q,
                types=req.types,
                k=self.settings.RELATED_K_DENSE_DEFAULT,
                exclude_ids=req.exclude_ids,
            )
            sparse_task = self.cms_client.knn_sparse(
                sparse_q,
                types=req.types,
                k=self.settings.RELATED_K_SPARSE_DEFAULT,
                exclude_ids=req.exclude_ids,
            )
            dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)

            fused = self._rrf_fuse(dense_hits, sparse_hits, k=self.settings.RRF_K)

        related_requests_total.labels(status="success").inc()

        # Gauge tracks how often dense + sparse agree — high overlap means
        # the two modes are redundant for this corpus; low overlap means
        # hybrid is pulling its weight.
        if fused:
            overlap = sum(1 for r in fused if len(r.sources) == 2)
            rrf_fusion_overlap_ratio.set(overlap / len(fused))

        logger.info(
            "related_complete",
            anchor=("content_id" if req.content_id else "text"),
            dense_hits=len(dense_hits),
            sparse_hits=len(sparse_hits),
            fused_total=len(fused),
            returned=min(len(fused), req.k),
        )
        return RelatedResponse(results=fused[: req.k])

    async def _resolve_query_vectors(
        self, req: RelatedRequest
    ) -> tuple[list[float], dict[str, float]]:
        """Return (dense_vec, sparse_map) for the query.

        - content_id path: read both from CMS (avoids re-embedding stored
          content; cheap; preserves the exact vectors used at write time).
        - text path: embed fresh via BGE-M3 (one forward pass = both modes).
        """
        if req.content_id:
            anchor = await self.cms_client.get_content_embeddings(req.content_id)
            dense = anchor.get("embedding")
            sparse = anchor.get("embedding_sparse")
            if not dense:
                raise ValueError(
                    f"content_id={req.content_id} has no dense embedding stored"
                )
            if not sparse:
                # Backfill not done for this item — degrade to dense-only by
                # passing an empty sparse map. CMS will short-circuit the
                # sparse kNN to 0 hits and RRF falls through to dense-only.
                logger.warning(
                    "related_anchor_missing_sparse",
                    content_id=req.content_id,
                    hint="run admin re-embed trigger to populate embedding_sparse",
                )
                sparse = {}
            return dense, sparse

        # text path
        encoded = await asyncio.to_thread(self.embedder.encode, [req.text], True)
        return encoded["dense"][0], encoded["sparse"][0]

    @staticmethod
    def _rrf_fuse(
        dense_hits: list[dict],
        sparse_hits: list[dict],
        k: int = 60,
    ) -> list[RelatedItem]:
        """Reciprocal Rank Fusion.

        For each item, sum 1/(k + rank_r) across rankings it appears in.
        Higher fused score = better. Items appearing in BOTH rankings get a
        natural boost since they accumulate two terms.

        k is the RRF dampening constant — standard literature value is 60.
        Larger k flattens the curve (later ranks count more); smaller k
        concentrates score in the top few. Tunable via env (RRF_K).
        """
        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, set[str]] = defaultdict(set)
        meta: dict[str, dict] = {}

        for rank, hit in enumerate(dense_hits):
            cid = hit["id"]
            scores[cid] += 1.0 / (k + rank + 1)
            sources[cid].add("dense")
            meta[cid] = hit  # keep at least one copy for type lookup

        for rank, hit in enumerate(sparse_hits):
            cid = hit["id"]
            scores[cid] += 1.0 / (k + rank + 1)
            sources[cid].add("sparse")
            meta.setdefault(cid, hit)

        # Sort fused descending. Stable tiebreak by content_id for
        # deterministic test assertions.
        ranked = sorted(
            scores.items(), key=lambda kv: (-kv[1], kv[0])
        )

        return [
            RelatedItem(
                content_id=cid,
                score=round(score, 6),
                content_type=meta[cid].get("type"),
                sources=sorted(sources[cid]),
            )
            for cid, score in ranked
        ]
