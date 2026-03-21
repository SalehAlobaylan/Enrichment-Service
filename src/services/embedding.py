import asyncio

from src.clients.cms import CMSClient
from src.models.embedder import EmbedderWrapper
from src.schemas.embed import EmbedQueryResponse, EmbedResponse
from src.utils.logging import get_logger
from src.utils.metrics import embedding_duration, embeddings_total

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, embedder: EmbedderWrapper, cms_client: CMSClient):
        self.embedder = embedder
        self.cms_client = cms_client

    async def embed(
        self,
        texts: list[str],
        content_ids: list[str] | None = None,
    ) -> EmbedResponse:
        with embedding_duration.time():
            vectors = await asyncio.to_thread(self.embedder.encode, texts)

        embeddings_total.labels(status="success").inc()

        response = EmbedResponse(
            embeddings=vectors,
            model=self.embedder.model_name,
            dimensions=self.embedder.dimensions,
        )

        if content_ids:
            await self._write_back(content_ids, vectors)

        return response

    async def embed_query(self, text: str) -> EmbedQueryResponse:
        vectors = await asyncio.to_thread(self.embedder.encode, [text])

        return EmbedQueryResponse(
            embedding=vectors[0],
            model=self.embedder.model_name,
            dimensions=self.embedder.dimensions,
        )

    async def _write_back(self, content_ids: list[str], vectors: list[list[float]]) -> None:
        for content_id, vector in zip(content_ids, vectors):
            try:
                await self.cms_client.store_embedding(content_id, vector)
                logger.info("embedding_writeback_complete", content_id=content_id)
            except Exception as exc:
                logger.error(
                    "embedding_writeback_failed",
                    content_id=content_id,
                    error=str(exc),
                )
