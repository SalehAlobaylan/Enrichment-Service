import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.config import Settings
from src.models.embedder import EmbedderWrapper
from src.models.whisper import WhisperWrapper
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._executor = ThreadPoolExecutor(max_workers=2)

        self.whisper = WhisperWrapper(
            model_size=settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            download_root=settings.MODELS_DIR,
        )
        self.embedder = EmbedderWrapper(
            model_name=settings.EMBEDDING_MODEL,
            cache_folder=settings.MODELS_DIR,
        )

    @property
    def is_ready(self) -> dict[str, bool]:
        return {
            "whisper": self.whisper.is_loaded,
            "embedder": self.embedder.is_loaded,
        }

    @property
    def all_ready(self) -> bool:
        return self.whisper.is_loaded and self.embedder.is_loaded

    async def warmup(self) -> None:
        loop = asyncio.get_event_loop()

        logger.info("loading_models")

        whisper_task = loop.run_in_executor(self._executor, self.whisper.load)
        embedder_task = loop.run_in_executor(self._executor, self.embedder.load)

        results = await asyncio.gather(whisper_task, embedder_task, return_exceptions=True)

        for name, result in zip(["whisper", "embedder"], results):
            if isinstance(result, Exception):
                logger.error("model_load_failed", model=name, error=str(result))
            else:
                logger.info("model_loaded", model=name)
