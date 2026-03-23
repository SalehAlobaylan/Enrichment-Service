from src.utils.logging import get_logger

logger = get_logger(__name__)

MAX_TEXT_LENGTH = 8192


class EmbedderWrapper:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_folder: str = "./models",
    ) -> None:
        self._model_name = model_name
        self._cache_folder = cache_folder
        self._model = None
        self._dimensions: int = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info("loading_embedder", model_name=self._model_name)
        self._model = SentenceTransformer(
            self._model_name,
            cache_folder=self._cache_folder,
        )
        test_embedding = self._model.encode(["test"])
        self._dimensions = len(test_embedding[0])
        logger.info(
            "embedder_loaded",
            model_name=self._model_name,
            dimensions=self._dimensions,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("Embedding model is not loaded. Call load() first.")

        truncated = [text[:MAX_TEXT_LENGTH] for text in texts]
        embeddings = self._model.encode(truncated, normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]
