"""Cross-encoder reranker — BAAI/bge-reranker-v2-m3 via FlagEmbedding.

A reranker is a cross-encoder: it takes a (query, candidate) PAIR as input
and produces a relevance score. This is dramatically more accurate than the
bi-encoder cosine similarity used during retrieval, because the model sees
both texts together and can attend across them. The cost is that you must
run inference on every pair you want to score — so rerankers are only
practical on a SMALL candidate set (typically top-K after retrieval).

Standard IR pattern:
  cheap retrieval (kNN) → top-50 candidates
  Dense kNN               → top-30 candidates
  expensive rerank       → top-10 candidates (final order)

bge-reranker-v2-m3 is the BAAI reranker paired with BGE-M3 (same training
data, same multilingual coverage — Arabic + English first-class). It exposes
the same `use_fp16` / cache_dir API as BGEM3FlagModel, so the wrapper shape
mirrors EmbedderWrapper exactly.

Memory footprint with `use_fp16=True`:
- Reranker fp16: ~3 GB loaded (similar to BGE-M3). Both together = ~6 GB.
- Cold start on CPU: ~30-60s, parallelized with BGE-M3 load via ModelManager.
"""
from src.common.utils.logging import get_logger

logger = get_logger(__name__)

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class RerankerWrapper:
    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        cache_folder: str = "./models",
        use_fp16: bool = True,
    ) -> None:
        self._model_name = model_name
        self._cache_folder = cache_folder
        self._use_fp16 = use_fp16
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        from FlagEmbedding import FlagReranker

        logger.info(
            "loading_reranker",
            model_name=self._model_name,
            use_fp16=self._use_fp16,
        )
        self._model = FlagReranker(
            self._model_name,
            use_fp16=self._use_fp16,
            cache_dir=self._cache_folder,
        )

        # Probe with an Arabic + English pair to verify multilingual coverage
        # and JIT-compile the model graph before serving traffic.
        self._model.compute_score(
            [
                ["test query", "test candidate"],
                ["استعلام", "مرشح"],
            ],
            normalize=True,
        )
        logger.info("reranker_loaded", model_name=self._model_name)

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Score each candidate against the query.

        Returns sigmoid-normalized scores in [0, 1]. Higher is better.
        Empty `candidates` returns an empty list (don't bother the model).
        """
        if self._model is None:
            raise RuntimeError("Reranker model is not loaded. Call load() first.")
        if not candidates:
            return []
        pairs = [[query, c] for c in candidates]
        scores = self._model.compute_score(pairs, normalize=True)
        # FlagReranker returns float when len(pairs)==1, list otherwise. Normalize.
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(s) for s in scores]
