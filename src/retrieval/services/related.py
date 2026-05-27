"""Hybrid retrieval orchestration for POST /v1/related.

Pipeline:
  1. Resolve query vector — either fetch (dense, sparse) for `content_id`
     from CMS, or embed `text` via BGE-M3 (one forward pass returns both).
  2. Fan out two parallel kNN queries against CMS (dense vs sparse), using
     candidate sizes from RELATED_K_*_DEFAULT.
  3. Fuse the two rankings with Reciprocal Rank Fusion (RRF) — no weights,
     no calibration, robust across query types and score scales.
  4. (Slice B) Rerank the top RERANK_INPUT_K candidates with a cross-encoder
     against the anchor text; reorder by rerank score. Skipped when
     req.rerank=False or when no anchor text is available.
  5. Truncate to the caller's `k` and return with per-item provenance.

Why RRF: the dense column uses cosine distance and the sparse column uses
negative inner product — different scales entirely. RRF only looks at RANK,
not raw scores, so cross-mode score normalization is unnecessary. k=60 is
the standard literature value (Cormack, Clarke, Büttcher 2009).

Why rerank: RRF is rank-based, not relevance-based. The top items after
RRF are "high in both rankings" — not necessarily "most relevant to the
query." Cross-encoders see both texts together and produce calibrated
relevance, dramatically better quality on the small candidate set RRF
hands them.
"""
import asyncio
from collections import defaultdict

from src.common.clients.cms import CMSClient
from src.common.config import Settings
from src.common.utils.logging import get_logger
from src.common.utils.metrics import (
    related_duration,
    related_requests_total,
    rerank_duration,
    rerank_requests_total,
    rrf_fusion_overlap_ratio,
)
from src.retrieval.models.embedder import EmbedderWrapper
from src.retrieval.models.reranker import RerankerWrapper
from src.retrieval.schemas.related import RelatedItem, RelatedRequest, RelatedResponse

logger = get_logger(__name__)


class RelatedService:
    def __init__(
        self,
        embedder: EmbedderWrapper,
        cms_client: CMSClient,
        settings: Settings,
        reranker: RerankerWrapper | None = None,
    ) -> None:
        self.embedder = embedder
        self.cms_client = cms_client
        self.settings = settings
        # Reranker is optional — None means rerank stage is unavailable
        # (e.g., model not loaded yet during cold start). All rerank=True
        # requests still succeed; they just fall through to RRF order.
        self.reranker = reranker

    async def related(self, req: RelatedRequest) -> RelatedResponse:
        with related_duration.time():
            dense_q, sparse_q, anchor_text = await self._resolve_query(req)

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

            # Gauge tracks how often dense + sparse agree — high overlap
            # means the two modes are redundant for this corpus; low overlap
            # means hybrid is pulling its weight.
            if fused:
                overlap = sum(1 for r in fused if len(r.sources) == 2)
                rrf_fusion_overlap_ratio.set(overlap / len(fused))

            # Slice B — rerank stage. Skipped when the caller disabled it,
            # the reranker model isn't loaded, or we don't have an anchor
            # text to compare against (e.g., rare content_id with no fields).
            reranked = fused
            if req.rerank and self.reranker and self.reranker.is_loaded and anchor_text:
                reranked = await self._rerank_candidates(
                    anchor_text, fused[: self.settings.RERANK_INPUT_K]
                )

        related_requests_total.labels(status="success").inc()
        logger.info(
            "related_complete",
            anchor=("content_id" if req.content_id else "text"),
            dense_hits=len(dense_hits),
            sparse_hits=len(sparse_hits),
            fused_total=len(fused),
            reranked=req.rerank
            and self.reranker is not None
            and self.reranker.is_loaded
            and bool(anchor_text),
            returned=min(len(reranked), req.k),
        )
        return RelatedResponse(results=reranked[: req.k])

    async def _resolve_query(
        self, req: RelatedRequest
    ) -> tuple[list[float], dict[str, float], str | None]:
        """Return (dense_vec, sparse_map, anchor_text_for_reranker).

        - content_id path: read vectors from CMS storage (avoid re-embedding;
          preserve exact write-time vectors). Anchor text is also fetched
          for the reranker stage to score (title + excerpt + body chunk).
        - text path: embed the raw text; that same text is the anchor for
          the reranker.
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
                # passing an empty sparse map. CMS short-circuits the sparse
                # kNN to 0 hits and RRF falls through to dense-only.
                logger.warning(
                    "related_anchor_missing_sparse",
                    content_id=req.content_id,
                    hint="run admin re-embed trigger to populate embedding_sparse",
                )
                sparse = {}
            # Anchor text for reranker: fetched on-demand from batch_text
            # in _rerank_candidates so we don't pay the cost when rerank
            # is disabled. Return a sentinel marker (the content_id itself)
            # that the rerank path interprets correctly.
            return dense, sparse, f"__cid__:{req.content_id}"

        # text path — caller supplied raw text; use it for both embedding
        # AND reranking.
        encoded = await asyncio.to_thread(self.embedder.encode, [req.text], True)
        return encoded["dense"][0], encoded["sparse"][0], req.text

    async def _rerank_candidates(
        self, anchor_text: str, candidates: list[RelatedItem]
    ) -> list[RelatedItem]:
        """Score (anchor, candidate) pairs with the cross-encoder, reorder.

        Resolves candidate text via CMS batch_text. Items missing from the
        batch response (e.g., deleted between kNN and batch fetch) drop out
        of the reranked list — they're already gone from CMS anyway.
        """
        if not candidates:
            return candidates

        candidate_ids = [c.content_id for c in candidates]
        items_by_id: dict[str, dict] = {}
        try:
            batch = await self.cms_client.batch_text(candidate_ids)
            items_by_id = {item["id"]: item for item in batch}
        except Exception as exc:
            # CMS unavailable — fall back to RRF order. The reranker stage
            # is enrichment, not a hard requirement; never fail the request.
            logger.warning("rerank_batch_text_failed", error=str(exc))
            rerank_requests_total.labels(status="failure").inc()
            return candidates

        # If the anchor came from a content_id, fetch its text too (the
        # batch endpoint accepts it). Sentinel-encoded in _resolve_query.
        if anchor_text.startswith("__cid__:"):
            cid = anchor_text.split(":", 1)[1]
            try:
                anchor_batch = await self.cms_client.batch_text([cid])
            except Exception as exc:
                logger.warning("rerank_anchor_text_fetch_failed", error=str(exc))
                rerank_requests_total.labels(status="failure").inc()
                return candidates
            if not anchor_batch:
                logger.warning("rerank_anchor_text_missing", content_id=cid)
                return candidates
            anchor_text = self._build_text_for_rerank(anchor_batch[0])

        # Build (query, candidate_text) pairs only for items we got text for.
        pair_items: list[RelatedItem] = []
        candidate_texts: list[str] = []
        for c in candidates:
            item = items_by_id.get(c.content_id)
            if item is None:
                continue
            text = self._build_text_for_rerank(item)
            if not text:
                continue
            pair_items.append(c)
            candidate_texts.append(text)

        if not pair_items:
            return candidates

        with rerank_duration.time():
            scores = await asyncio.to_thread(
                self.reranker.rerank, anchor_text, candidate_texts
            )
        rerank_requests_total.labels(status="success").inc()

        # Build new ordered list: reranked items first (by rerank score
        # desc, stable tie-break by content_id); any items that didn't get
        # reranked tacked on at the end in their RRF order — they shouldn't
        # outrank anything the reranker actually scored.
        reranked: list[RelatedItem] = []
        ranked_ids: set[str] = set()
        for c, score in zip(pair_items, scores):
            item = items_by_id[c.content_id]
            reranked.append(
                RelatedItem(
                    content_id=c.content_id,
                    score=round(float(score), 6),
                    content_type=c.content_type,
                    sources=c.sources,
                    rerank_score=round(float(score), 6),
                    published_at=item.get("published_at"),
                    source_name=item.get("source_name"),
                )
            )
            ranked_ids.add(c.content_id)

        reranked.sort(key=lambda r: (-r.score, r.content_id))

        # Items dropped from the rerank pool — keep them in RRF order as
        # tail candidates. (Typically zero items, but defensive.)
        tail = [c for c in candidates if c.content_id not in ranked_ids]
        return reranked + tail

    @staticmethod
    def _build_text_for_rerank(item: dict) -> str:
        """Construct the text passed to the cross-encoder.

        Priority: title + excerpt (compact, high-signal for ARTICLE) → body
        truncated to first ~512 chars (for TWEET/COMMENT where body IS the
        signal). Empty when all three fields are missing.
        """
        title = (item.get("title") or "").strip()
        excerpt = (item.get("excerpt") or "").strip()
        body = (item.get("body_text") or "").strip()

        parts: list[str] = []
        if title:
            parts.append(title)
        if excerpt:
            parts.append(excerpt)
        if not parts and body:
            parts.append(body[:512])
        return " ".join(parts)

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
                # Hydrate ranking-rule inputs from the kNN response so
                # freshness decay + source diversity still work when the
                # rerank stage is skipped (cold start, RERANK_ENABLED=False,
                # or no anchor text). Reranker overwrites these later with
                # the same values from batch_text — no behavioral change
                # to the rerank path.
                published_at=meta[cid].get("published_at"),
                source_name=meta[cid].get("source_name"),
            )
            for cid, score in ranked
        ]
