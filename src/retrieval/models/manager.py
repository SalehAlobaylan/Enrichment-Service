"""Model manager for Enrichment-Service.

After Slice 0 + Slice B, this service hosts two text-side models:
  - Embedder:  BGE-M3 (dense + sparse) for retrieval kNN.
  - Reranker:  bge-reranker-v2-m3 cross-encoder for refining top candidates.

Both load in parallel at startup. Whisper transcription + CLIP image
embedding live in Media-Service.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.common.config import Settings
from src.common.utils.logging import get_logger
from src.retrieval.models.embedder import EmbedderWrapper
from src.retrieval.models.reranker import RerankerWrapper

logger = get_logger(__name__)


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Two concurrent loaders — cold start is bottlenecked on the slowest
        # of the two (typically the embedder; reranker is a bit faster).
        self._executor = ThreadPoolExecutor(max_workers=2)

        self.embedder = EmbedderWrapper(
            model_name=settings.EMBEDDING_MODEL,
            cache_folder=settings.MODELS_DIR,
        )
        self.reranker = RerankerWrapper(
            model_name=settings.RERANKER_MODEL,
            cache_folder=settings.MODELS_DIR,
        )

    @property
    def is_ready(self) -> dict[str, bool]:
        return {
            "embedder": self.embedder.is_loaded,
            "reranker": self.reranker.is_loaded,
        }

    @property
    def all_ready(self) -> bool:
        # Reranker is best-effort: /v1/related and /v1/feed/news/slide both
        # degrade gracefully to RRF order when it isn't loaded. Gating
        # /ready on the reranker caused Cranl health probes to mark the pod
        # unhealthy during the ~60s window between embedder-loaded and
        # reranker-loaded, triggering restart loops on cold start.
        return self.embedder.is_loaded

    async def warmup(self) -> None:
        loop = asyncio.get_event_loop()

        logger.info("loading_models")

        embedder_task = loop.run_in_executor(self._executor, self.embedder.load)
        reranker_task = loop.run_in_executor(self._executor, self.reranker.load)

        results = await asyncio.gather(
            embedder_task, reranker_task, return_exceptions=True
        )

        for name, result in zip(["embedder", "reranker"], results):
            if isinstance(result, Exception):
                logger.error("model_load_failed", model=name, error=str(result))
            else:
                logger.info("model_loaded", model=name)
