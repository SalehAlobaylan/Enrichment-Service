"""Text embedder wrapper — BAAI/bge-m3 dense + sparse via FlagEmbedding.

BGE-M3 is multilingual (Arabic + English first-class) and produces two
complementary outputs from one forward pass:
  - Dense (1024-dim float vector): semantic similarity. Catches paraphrase.
  - Sparse (learned lexical weights over a 250002-token vocab): exact-name
    + morphology matching. Strictly better than BM25 for Arabic.

Both are needed for hybrid retrieval (Slice A). FlagEmbedding's
BGEM3FlagModel is the official BAAI wrapper that exposes both modes; it
uses sentence-transformers under the hood for the dense path, so there's
no duplicate model load.

Memory footprint with `use_fp16=True`:
- BGE-M3 fp16: ~3 GB loaded (vs ~6 GB fp32) — negligible quality loss for retrieval
- Cold start on CPU: ~60-90s
"""
from src.common.utils.logging import get_logger

logger = get_logger(__name__)

# BGE-M3 was trained on inputs up to 8192 tokens — well beyond what
# Wahb's content items will ever feed it. The cap stays as a defensive
# truncation for malformed input.
MAX_TEXT_LENGTH = 8192

# BGE-M3 vocabulary size — matches the `sparsevec(250002)` column type in
# the CMS schema (migration 20260522000000_bge_m3_retrieval.sql).
BGE_M3_SPARSE_DIM = 250002


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
    def sparse_dim(self) -> int:
        """Vocabulary size of BGE-M3's sparse output. Matches sparsevec(N) in CMS."""
        return BGE_M3_SPARSE_DIM

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        from FlagEmbedding import BGEM3FlagModel

        logger.info(
            "loading_embedder",
            model_name=self._model_name,
            use_fp16=self._use_fp16,
            backend="BGEM3FlagModel",
        )
        self._model = BGEM3FlagModel(
            self._model_name,
            use_fp16=self._use_fp16,
            cache_dir=self._cache_folder,
        )

        # Probe with an Arabic + English mix to confirm tokenizer handles
        # non-Latin scripts (the whole point of switching to BGE-M3) and to
        # bind dense dimensions before serving traffic.
        probe = self._model.encode(
            ["test مرحبا"],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        self._dimensions = len(probe["dense_vecs"][0])
        logger.info(
            "embedder_loaded",
            model_name=self._model_name,
            dimensions=self._dimensions,
            sparse_dim=BGE_M3_SPARSE_DIM,
        )

    def encode(
        self,
        texts: list[str],
        with_sparse: bool = True,
    ) -> dict[str, list]:
        """Encode texts to dense (+ optional sparse) vectors.

        Returns:
            {
                "dense":  list[list[float]],            # always (n_texts, 1024)
                "sparse": list[dict[str, float]] | None # token_id_str → weight, or None
            }

        with_sparse=False is for read-only paths like /v1/embed/query where
        only the dense vector matters; saves a small amount of compute.
        """
        if self._model is None:
            raise RuntimeError("Embedding model is not loaded. Call load() first.")

        truncated = [text[:MAX_TEXT_LENGTH] for text in texts]
        out = self._model.encode(
            truncated,
            return_dense=True,
            return_sparse=with_sparse,
            return_colbert_vecs=False,
        )
        # BGE-M3 returns numpy arrays for dense and list[dict] for sparse
        # (lexical_weights). With use_fp16=True the sparse WEIGHTS are numpy
        # float16 scalars and the keys are numpy ints — neither is JSON
        # serializable, so the CMS write-back (json.dumps) fails with
        # "Object of type float16 is not JSON serializable". `.tolist()` already
        # coerces dense to Python floats; the sparse dict must be coerced
        # explicitly (keys → str, weights → float).
        dense = [d.tolist() for d in out["dense_vecs"]]
        sparse = (
            [
                {str(token): float(weight) for token, weight in weights.items()}
                for weights in out["lexical_weights"]
            ]
            if with_sparse
            else None
        )
        return {"dense": dense, "sparse": sparse}
