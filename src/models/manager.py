"""Model manager for Enrichment-Service.

Loads the text embedder. Whisper transcription and CLIP image embedding
have moved to Media-Service (their own dedicated service). The text
embedder will be swapped to BGE-M3 in Slice 0; this manager will then
also load the reranker.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.config import Settings
from src.models.embedder import EmbedderWrapper
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Single-loader executor for now. Slice 0 grows this to 2 (embedder
        # + reranker) so cold start can be parallelized.
        self._executor = ThreadPoolExecutor(max_workers=2)

        self.embedder = EmbedderWrapper(
            model_name=settings.EMBEDDING_MODEL,
            cache_folder=settings.MODELS_DIR,
        )

    @property
    def is_ready(self) -> dict[str, bool]:
        return {
            "embedder": self.embedder.is_loaded,
        }

    @property
    def all_ready(self) -> bool:
        return self.embedder.is_loaded

    async def warmup(self) -> None:
        loop = asyncio.get_event_loop()

        logger.info("loading_models")

        embedder_task = loop.run_in_executor(self._executor, self.embedder.load)

        results = await asyncio.gather(embedder_task, return_exceptions=True)

        for name, result in zip(["embedder"], results):
            if isinstance(result, Exception):
                logger.error("model_load_failed", model=name, error=str(result))
            else:
                logger.info("model_loaded", model=name)
