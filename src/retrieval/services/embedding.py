import asyncio

from src.common.clients.cms import CMSClient
from src.common.utils.logging import get_logger
from src.common.utils.metrics import embedding_duration, embeddings_total
from src.llm.services.tagging import TaggingService, TagsResult
from src.retrieval.models.embedder import EmbedderWrapper
from src.retrieval.schemas.embed import EmbedQueryResponse, EmbedResponse

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(
        self,
        embedder: EmbedderWrapper,
        cms_client: CMSClient,
        tagger: TaggingService | None = None,
    ):
        self.embedder = embedder
        self.cms_client = cms_client
        # Tagger is optional so EmbeddingService stays usable in stateless
        # tool mode (no LLM configured). When None, extract_tags=True is
        # silently ignored.
        self.tagger = tagger

    async def embed(
        self,
        texts: list[str],
        content_ids: list[str] | None = None,
        extract_tags: bool = False,
    ) -> EmbedResponse:
        # Run embedding compute and tag-extraction concurrently when tagging
        # is requested — embedder is CPU, tagging is network-bound on the LLM.
        # Tagging operates on the FIRST text (caller intent: a single content
        # item, optionally batched by chunk).
        tags_task: asyncio.Task[TagsResult] | None = None
        if extract_tags and self.tagger and texts:
            tags_task = asyncio.create_task(self.tagger.extract(texts[0]))

        with embedding_duration.time():
            vectors = await asyncio.to_thread(self.embedder.encode, texts)

        embeddings_total.labels(status="success").inc()

        tags_result: TagsResult | None = None
        if tags_task is not None:
            try:
                tags_result = await tags_task
            except Exception as exc:
                logger.warning("tag_extraction_task_failed", error=str(exc))

        response = EmbedResponse(
            embeddings=vectors,
            model=self.embedder.model_name,
            dimensions=self.embedder.dimensions,
        )
        if tags_result is not None:
            response.tags = tags_result.tags
            response.entities = tags_result.entities

        if content_ids:
            topic_tags = tags_result.tags if tags_result else None
            status, error = await self._write_back(content_ids, vectors, topic_tags)
            response.write_back_status = status
            response.write_back_error = error

        return response

    async def embed_query(self, text: str) -> EmbedQueryResponse:
        vectors = await asyncio.to_thread(self.embedder.encode, [text])

        return EmbedQueryResponse(
            embedding=vectors[0],
            model=self.embedder.model_name,
            dimensions=self.embedder.dimensions,
        )

    async def _write_back(
        self,
        content_ids: list[str],
        vectors: list[list[float]],
        topic_tags: list[str] | None = None,
    ) -> tuple[str, str | None]:
        """Write each (content_id, vector) to CMS. Returns (status, first_error).

        status is "ok" only if every write succeeded; "failed" if any did. We
        still attempt every write so partial successes get persisted — the
        returned error string is the first failure encountered, which is
        enough to alert callers without flooding the response.

        topic_tags, when provided, is sent on every write — the assumption is
        a batch of `texts` represents chunks of the same conceptual item.
        """
        first_error: str | None = None
        for content_id, vector in zip(content_ids, vectors):
            try:
                await self.cms_client.store_embedding(
                    content_id, vector, topic_tags=topic_tags
                )
                logger.info(
                    "embedding_writeback_complete",
                    content_id=content_id,
                    topic_tag_count=len(topic_tags) if topic_tags else 0,
                )
            except Exception as exc:
                err = str(exc)
                logger.error(
                    "embedding_writeback_failed",
                    content_id=content_id,
                    error=err,
                )
                if first_error is None:
                    first_error = err

        return ("failed", first_error) if first_error else ("ok", None)
