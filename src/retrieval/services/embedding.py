import asyncio

from src.common.clients.cms import CMSClient
from src.common.utils.logging import get_logger
from src.common.utils.metrics import embedding_duration, embeddings_total
from src.common.workload_admission import WorkloadExecutors
from src.llm.services.tagging import TaggingService, TagsResult
from src.retrieval.models.embedder import EmbedderWrapper
from src.retrieval.schemas.embed import EmbedQueryResponse, EmbedResponse

logger = get_logger(__name__)

WRITEBACK_CONCURRENCY = 4


def _space_descriptor(embedder: EmbedderWrapper) -> dict:
    """Return a real descriptor, tolerating legacy/test embedder doubles."""
    factory = getattr(embedder, "space_descriptor", None)
    if not callable(factory):
        return {}
    descriptor = factory()
    return descriptor if isinstance(descriptor, dict) else {}


class EmbeddingService:
    def __init__(
        self,
        embedder: EmbedderWrapper,
        cms_client: CMSClient,
        tagger: TaggingService | None = None,
        executors: WorkloadExecutors | None = None,
    ):
        self.embedder = embedder
        self.cms_client = cms_client
        # Tagger is optional so EmbeddingService stays usable in stateless
        # tool mode (no LLM configured). When None, extract_tags=True is
        # silently ignored.
        self.tagger = tagger
        self.executors = executors

    async def _encode(self, texts: list[str], extract_sparse: bool):
        if self.executors is not None:
            return await self.executors.run(
                "embedding", self.embedder.encode, texts, extract_sparse
            )
        return await asyncio.to_thread(self.embedder.encode, texts, extract_sparse)

    async def embed(
        self,
        texts: list[str],
        content_ids: list[str] | None = None,
        extract_tags: bool = False,
        extract_sparse: bool = False,
        artifact_recovery: dict[str, str] | None = None,
        pipeline_repair: dict[str, str] | None = None,
    ) -> EmbedResponse:
        # Run embedding compute and tag-extraction concurrently when tagging
        # is requested — embedder is CPU, tagging is network-bound on the LLM.
        # Tagging operates on the FIRST text (caller intent: a single content
        # item, optionally batched by chunk).
        tags_task: asyncio.Task[TagsResult] | None = None
        if extract_tags and self.tagger and texts:
            tags_task = asyncio.create_task(self.tagger.extract(texts[0]))

        # Qwen is dense-only — `extract_sparse` is a no-op kept for call-site
        # compatibility; the embedder always returns sparse=None and the CMS
        # `embedding_sparse` column stays NULL.
        with embedding_duration.time():
            encoded = await self._encode(texts, extract_sparse)
        dense_vectors: list[list[float]] = encoded["dense"]
        sparse_maps: list[dict[str, float]] | None = encoded["sparse"]

        embeddings_total.labels(status="success").inc()

        tags_result: TagsResult | None = None
        if tags_task is not None:
            try:
                tags_result = await tags_task
            except Exception as exc:
                logger.warning("tag_extraction_task_failed", error=str(exc))

        descriptor = _space_descriptor(self.embedder)
        response = EmbedResponse(
            embeddings=dense_vectors,
            model=self.embedder.model_name,
            dimensions=self.embedder.dimensions,
            space_id=descriptor.get("space_id", ""),
            producer_id=descriptor.get("producer_id", ""),
        )
        if tags_result is not None:
            response.tags = tags_result.tags
            response.entities = tags_result.entities

        if content_ids:
            topic_tags = tags_result.tags if tags_result else None
            status, error = await self._write_back(
                content_ids,
                dense_vectors,
                sparse_maps,
                topic_tags,
                descriptor,
                artifact_recovery,
                pipeline_repair,
            )
            response.write_back_status = status
            response.write_back_error = error

        return response

    async def embed_query(self, text: str) -> EmbedQueryResponse:
        # Query embedding path only needs dense — sparse is for write-back to
        # CMS, not for ad-hoc query embedding (callers use /v1/related for
        # dense query search, which embeds query text separately).
        encoded = await self._encode([text], False)

        descriptor = _space_descriptor(self.embedder)
        return EmbedQueryResponse(
            embedding=encoded["dense"][0],
            model=self.embedder.model_name,
            dimensions=self.embedder.dimensions,
            space_id=descriptor.get("space_id", ""),
            producer_id=descriptor.get("producer_id", ""),
        )

    async def _write_back(
        self,
        content_ids: list[str],
        vectors: list[list[float]],
        sparse_maps: list[dict[str, float]] | None,
        topic_tags: list[str] | None = None,
        descriptor: dict | None = None,
        artifact_recovery: dict[str, str] | None = None,
        pipeline_repair: dict[str, str] | None = None,
    ) -> tuple[str, str | None]:
        """Write each (content_id, vector, [sparse]) to CMS.

        Returns (status, first_error). status is "ok" only if every write
        succeeded; "failed" if any did. We still attempt every write so
        partial successes get persisted — the returned error string is the
        first failure encountered, which is enough to alert callers without
        flooding the response.

        topic_tags, when provided, is sent on every write — the assumption is
        a batch of `texts` represents chunks of the same conceptual item.

        sparse_maps, when None, means the caller didn't request sparse
        extraction; the CMS column stays NULL (forward-compatible with the
        Week 2b sparsevec column).
        """
        if len(content_ids) != len(vectors):
            logger.error(
                "embedding_writeback_length_mismatch",
                content_id_count=len(content_ids),
                vector_count=len(vectors),
            )
            return "failed", "writeback_length_mismatch"
        if sparse_maps is not None and len(sparse_maps) != len(vectors):
            logger.error(
                "embedding_writeback_sparse_length_mismatch",
                sparse_count=len(sparse_maps),
                vector_count=len(vectors),
            )
            return "failed", "writeback_sparse_length_mismatch"

        semaphore = asyncio.Semaphore(WRITEBACK_CONCURRENCY)

        async def write_one(i: int, content_id: str, vector: list[float]) -> str | None:
            sparse = sparse_maps[i] if sparse_maps else None
            try:
                async with semaphore:
                    await self.cms_client.store_embedding(
                        content_id,
                        vector,
                        topic_tags=topic_tags,
                        embedding_sparse=sparse,
                        model=self.embedder.model_name,
                        space_id=descriptor.get("space_id") if descriptor else None,
                        producer_id=descriptor.get("producer_id") if descriptor else None,
                        artifact_recovery=artifact_recovery,
                        pipeline_repair=pipeline_repair,
                    )
                logger.info(
                    "embedding_writeback_complete",
                    content_id=content_id,
                    topic_tag_count=len(topic_tags) if topic_tags else 0,
                    has_sparse=sparse is not None,
                )
                return None
            except Exception:  # noqa: BLE001 - write-back is best effort
                # Do not return provider/CMS detail to callers. A stable value
                # preserves partial-outcome truth without leaking internals.
                err = "cms_writeback_failed"
                logger.error(
                    "embedding_writeback_failed",
                    content_id=content_id,
                    error=err,
                )
                return err

        errors = await asyncio.gather(
            *(
                write_one(i, content_id, vector)
                for i, (content_id, vector) in enumerate(zip(content_ids, vectors))
            )
        )
        first_error = next((error for error in errors if error is not None), None)
        return ("failed", first_error) if first_error else ("ok", None)
