"""Dense retrieval orchestration for POST /v1/related.

Pipeline:
  1. Resolve query vector — either fetch the dense vector for `content_id`
     from CMS, or embed `text` via Qwen3-Embedding-0.6B.
  2. Query dense kNN against CMS's matching immutable vector space.
  3. Rerank the top RERANK_INPUT_K candidates with a cross-encoder
     against the anchor text; reorder by rerank score. Skipped when
     req.rerank=False or when no anchor text is available.
  4. Truncate to the caller's `k` and return with per-item provenance.

Why rerank: nearest-neighbor order is similarity-based, not calibrated
relevance. Cross-encoders see both texts together and produce better quality
on the small candidate set passed to them.
"""
import asyncio

from src.common.clients.cms import CMSClient
from src.common.config import Settings
from src.common.utils.logging import get_logger
from src.common.utils.metrics import (
    related_duration,
    related_requests_total,
    rerank_duration,
    rerank_requests_total,
)
from src.common.workload_admission import WorkloadAdmission, WorkloadExecutors
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
        admission: WorkloadAdmission | None = None,
        executors: WorkloadExecutors | None = None,
    ) -> None:
        self.embedder = embedder
        self.cms_client = cms_client
        self.settings = settings
        # Reranker is optional — None means rerank stage is unavailable
        # (e.g., model not loaded yet during cold start). All rerank=True
        # requests still succeed; they just retain dense kNN order.
        self.reranker = reranker
        self.admission = admission
        self.executors = executors

    async def _run_blocking(self, workload: str, function, *args):
        if self.admission is None:
            if self.executors is not None:
                return await self.executors.run(workload, function, *args)
            return await asyncio.to_thread(function, *args)
        async with self.admission.acquire(workload):
            if self.executors is not None:
                return await self.executors.run(workload, function, *args)
            return await asyncio.to_thread(function, *args)

    async def related(self, req: RelatedRequest) -> RelatedResponse:
        with related_duration.time():
            dense_q, anchor_text, space_id = await self._resolve_query(req)

            dense_hits = await self.cms_client.knn_dense(
                dense_q,
                space_id,
                types=req.types,
                formats=req.formats,
                k=self.settings.RELATED_K_DENSE_DEFAULT,
                exclude_ids=req.exclude_ids,
            )
            candidates = self._dense_results(dense_hits)

            # Slice B — rerank stage. Skipped when the caller disabled it,
            # the reranker model isn't loaded, or we don't have an anchor
            # text to compare against (e.g., rare content_id with no fields).
            reranked = candidates
            if req.rerank and self.reranker and self.reranker.is_loaded and anchor_text:
                reranked = await self._rerank_candidates(
                    anchor_text, candidates[: self.settings.RERANK_INPUT_K]
                )

        related_requests_total.labels(status="success").inc()
        logger.info(
            "related_complete",
            anchor=("content_id" if req.content_id else "text"),
            dense_hits=len(dense_hits),
            candidates_total=len(candidates),
            reranked=req.rerank
            and self.reranker is not None
            and self.reranker.is_loaded
            and bool(anchor_text),
            returned=min(len(reranked), req.k),
        )
        return RelatedResponse(results=reranked[: req.k])

    async def _resolve_query(
        self, req: RelatedRequest
    ) -> tuple[list[float], str | None, str]:
        """Return (dense_vec, anchor_text_for_reranker, space_id).

        - content_id path: read vectors from CMS storage (avoid re-embedding;
          preserve exact write-time vectors). Anchor text is also fetched
          for the reranker stage to score (title + excerpt + body chunk).
        - text path: embed the raw text; that same text is the anchor for
          the reranker.
        """
        if req.content_id:
            anchor = await self.cms_client.get_content_embeddings(req.content_id)
            dense = anchor.get("embedding")
            space_id = anchor.get("embedding_space_id")
            if not dense:
                raise ValueError(
                    f"content_id={req.content_id} has no dense embedding stored"
                )
            if not space_id:
                raise ValueError(
                    f"content_id={req.content_id} has an unstamped dense embedding"
                )
            # Anchor text for reranker: fetched on-demand from batch_text
            # in _rerank_candidates so we don't pay the cost when rerank
            # is disabled. Return a sentinel marker (the content_id itself)
            # that the rerank path interprets correctly.
            return dense, f"__cid__:{req.content_id}", space_id

        # text path — caller supplied raw text; use it for both embedding
        # AND reranking.
        encoded = await self._run_blocking("embedding", self.embedder.encode, [req.text], False)
        raw_desc = self.embedder.space_descriptor()
        desc = raw_desc if isinstance(raw_desc, dict) else {}
        space_id = desc.get("space_id", "")
        if not space_id:
            raise ValueError("query embedding space is unresolved")
        return encoded["dense"][0], req.text, space_id

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
            # CMS unavailable — fall back to dense kNN order. The reranker stage
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

        # Degrade gracefully on any rerank failure: reranking is enrichment,
        # not a hard requirement — fall back to dense kNN order rather than failing
        # the whole /related or news-slide request. This is a PERMANENT,
        # general safeguard (keep it whether or not the reranker is split out);
        # it also happens to cover the extra failure mode the TEMP split adds
        # (the reranker being a remote HTTP call — see ModelManager).
        try:
            with rerank_duration.time():
                scores = await self._run_blocking(
                    "rerank", self.reranker.rerank, anchor_text, candidate_texts
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("rerank_scoring_failed", error=str(exc))
            rerank_requests_total.labels(status="failure").inc()
            return candidates
        rerank_requests_total.labels(status="success").inc()

        # Build new ordered list: reranked items first (by rerank score
        # desc, stable tie-break by content_id); any items that didn't get
        # reranked tacked on at the end in their dense kNN order — they shouldn't
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
                    content_format=c.content_format,
                    sources=c.sources,
                    rerank_score=round(float(score), 6),
                    published_at=item.get("published_at"),
                    source_name=item.get("source_name"),
                )
            )
            ranked_ids.add(c.content_id)

        reranked.sort(key=lambda r: (-r.score, r.content_id))

        # Items dropped from the rerank pool — keep them in dense kNN order as
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
        # Cap the pair text — cross-encoder cost scales with sequence length,
        # and the split reranker runs on a constrained CPU. Title+excerpt is
        # plenty of signal; ~300 chars keeps each pair fast.
        return " ".join(parts)[:300]

    @staticmethod
    def _dense_results(dense_hits: list[dict]) -> list[RelatedItem]:
        """Map CMS dense kNN hits to the response contract without fusion.

        Qwen produces a single dense vector, so the CMS kNN order and score
        are the canonical retrieval provenance until the optional reranker
        replaces them. Duplicate IDs are ignored defensively while retaining
        the first (best-ranked) hit.
        """
        results: list[RelatedItem] = []
        seen: set[str] = set()
        for hit in dense_hits:
            content_id = str(hit.get("id") or "")
            if not content_id or content_id in seen:
                continue
            seen.add(content_id)
            results.append(
                RelatedItem(
                    content_id=content_id,
                    score=round(float(hit.get("score") or 0.0), 6),
                    content_type=hit.get("type"),
                    content_format=hit.get("format"),
                    sources=["dense"],
                    published_at=hit.get("published_at"),
                    source_name=hit.get("source_name"),
                )
            )
        return results
