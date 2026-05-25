"""Text embedder wrapper — BAAI/bge-m3 dense path.

BGE-M3 is multilingual (Arabic + English first-class) and produces 1024-dim
dense vectors. This wrapper exposes the dense path only via
sentence-transformers, which is already a dependency.

Sparse output (BGE-M3's lexical-weights mode for hybrid retrieval) is
intentionally NOT exposed here yet — Slice A will add the FlagEmbedding
library and a second `encode_sparse()` method. The CMS schema already has
the `embedding_sparse sparsevec(250002)` column standing by; until Slice A
populates it, the column stays NULL.

Memory footprint with `use_fp16=True`:
- BGE-M3 fp16: ~3 GB loaded (vs ~6 GB fp32) — negligible quality loss for retrieval
- Cold start on CPU: ~60-90s (vs ~10s for the old all-MiniLM-L6-v2)
"""
from src.common.utils.logging import get_logger

logger = get_logger(__name__)

# BGE-M3 was trained on inputs up to 8192 tokens — well beyond what
# Wahb's content items will ever feed it. The cap stays as a defensive
# truncation for malformed input.
MAX_TEXT_LENGTH = 8192


class EmbedderWrapper:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_folder: str = "./models",
        use_fp16: bool = True,
    ) -> None:
        self._model_name = model_name
        self._cache_folder = cache_folder
        self._use_fp16 = use_fp16
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

        logger.info(
            "loading_embedder",
            model_name=self._model_name,
            use_fp16=self._use_fp16,
        )
        self._model = SentenceTransformer(
            self._model_name,
            cache_folder=self._cache_folder,
        )
        if self._use_fp16:
            # Halves RAM (~6 GB → ~3 GB) with negligible quality loss for
            # retrieval. Skip on devices that don't support fp16 by setting
            # use_fp16=False (CPU supports it; ancient hardware may not).
            try:
                self._model.half()
            except Exception as exc:
                logger.warning(
                    "fp16_failed_falling_back_to_fp32",
                    error=str(exc),
                )

        # Probe with an Arabic + English token mix to confirm the tokenizer
        # handles non-Latin scripts (the whole point of switching to BGE-M3).
        test_embedding = self._model.encode(["test مرحبا"])
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
