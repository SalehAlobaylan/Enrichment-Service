"""Text embedder wrapper — Qwen/Qwen3-Embedding-0.6B (dense-only) via
sentence-transformers.

Qwen3-Embedding-0.6B is multilingual (strong Arabic + English) and produces a
single 1024-dim dense vector per input — no sparse/lexical output. Wahb moved
off BGE-M3's hybrid (dense + sparse) to Qwen dense-only: simpler, better Arabic
retrieval in practice, and one model load. The CMS `embedding_sparse` column is
left NULL (retired in a follow-up migration).

Memory footprint:
- Qwen3-Embedding-0.6B fp32: ~2.4 GB loaded; fp16 (GPU only) ~1.2 GB
- CPU cold start: ~20-40s
"""

import re
from pathlib import Path

from src.common.spaceid import (
    RECIPE_CONTENT_TEXT,
    compute_producer_id,
    compute_space_id,
)
from src.common.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_hf_revision(model_name: str, cache_folder: str, revision: str = "") -> str:
    """Resolve this exact selector to a snapshot, then load only that commit.

    Reading an arbitrary cache ref after loading cannot prove which weights were
    loaded. The config download resolves the selected branch/tag before loading;
    its immutable snapshot commit is passed to SentenceTransformer as well.
    Local/custom artifacts without this evidence remain lifecycle-not-ready.
    """
    from huggingface_hub import hf_hub_download

    repo = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    try:
        config_path = Path(
            hf_hub_download(
                repo,
                "config.json",
                cache_dir=cache_folder,
                revision=revision or None,
            )
        )
    except (OSError, ValueError):
        return ""
    for parent in config_path.parents:
        if parent.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", parent.name):
            return parent.name
    return ""


# Qwen3-Embedding-0.6B handles up to 32k tokens — far beyond any Wahb content
# item. The cap is a defensive char-truncation against malformed input.
MAX_TEXT_LENGTH = 8192


class EmbedderWrapper:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        cache_folder: str = "./models",
        use_fp16: bool = True,
        revision: str = "",
    ) -> None:
        self._model_name = model_name
        self._cache_folder = cache_folder
        self._use_fp16 = use_fp16
        self._revision_selector = revision.strip()
        self._revision = ""
        self._model = None
        self._dimensions: int = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def revision(self) -> str:
        """Immutable artifact digest, or "" when unresolved."""
        return self._revision

    def space_descriptor(self) -> dict:
        """Vector-space basis + producer identity for the lifecycle system.

        Returned as-is on /v1/models and used to stamp write-backs. space_id ==
        "" means the space is not lifecycle-ready (unresolved revision) and no
        write may be stamped with it.
        """
        space_id = compute_space_id(
            model=self._model_name,
            revision=self._revision,
            dimensions=self._dimensions,
            normalized=True,
            pooling="sentence-transformers-config",
        )
        return {
            "model": self._model_name,
            "revision": self._revision,
            "dimensions": self._dimensions,
            "normalized": True,
            "pooling": "sentence-transformers-config",
            "space_id": space_id,
            "producer_recipe": RECIPE_CONTENT_TEXT,
            "producer_id": compute_producer_id(space_id, RECIPE_CONTENT_TEXT),
        }

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def sparse_dim(self) -> int:
        """Qwen is dense-only — no sparse/lexical output. Kept for interface
        compatibility with the former BGE-M3 wrapper; always 0."""
        return 0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        import torch
        from sentence_transformers import SentenceTransformer

        logger.info(
            "loading_embedder",
            model_name=self._model_name,
            backend="SentenceTransformer",
        )
        resolved_revision = _resolve_hf_revision(
            self._model_name,
            self._cache_folder,
            self._revision_selector,
        )
        self._model = SentenceTransformer(
            self._model_name,
            cache_folder=self._cache_folder,
            revision=resolved_revision or self._revision_selector or None,
        )
        # fp16 only helps on GPU; PyTorch CPU lacks half-precision kernels for
        # many ops, so keep fp32 on CPU (Cranl is CPU-only).
        if self._use_fp16 and torch.cuda.is_available():
            self._model = self._model.half()

        # Probe with an Arabic + English mix to confirm the multilingual
        # tokenizer handles non-Latin scripts and to bind the dense dimension
        # before serving traffic.
        probe = self._model.encode(
            ["test مرحبا"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self._dimensions = len(probe[0])
        self._revision = resolved_revision
        logger.info(
            "embedder_loaded",
            model_name=self._model_name,
            dimensions=self._dimensions,
            revision=self._revision or "unresolved",
        )

    def encode(
        self,
        texts: list[str],
        with_sparse: bool = False,  # noqa: ARG002 — kept for call-site compat
    ) -> dict[str, list]:
        """Encode texts to L2-normalized dense vectors.

        Returns:
            {
                "dense":  list[list[float]],  # (n_texts, 1024), L2-normalized
                "sparse": None                # Qwen is dense-only
            }

        `with_sparse` is accepted for call-site compatibility with the former
        BGE-M3 wrapper but ignored — Qwen produces no sparse output.
        """
        if self._model is None:
            raise RuntimeError("Embedding model is not loaded. Call load() first.")

        truncated = [text[:MAX_TEXT_LENGTH] for text in texts]
        vectors = self._model.encode(
            truncated,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        dense = [v.tolist() for v in vectors]
        return {"dense": dense, "sparse": None}
