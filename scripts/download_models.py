"""Pre-download Enrichment-Service ML models for Docker image build.

Whisper + CLIP live in Media-Service's download script. This pre-caches the
two text-side models hosted by Enrichment:

  - Qwen/Qwen3-Embedding-0.6B:  multilingual text embedder (dense-only)
  - BAAI/bge-reranker-v2-m3:    cross-encoder reranker (Slice B)

Usage:
    python scripts/download_models.py [--output /path/to/models]
"""

import argparse
import os


def download_embedder(
    output_dir: str, model_name: str = "Qwen/Qwen3-Embedding-0.6B"
) -> None:
    # Pre-cache via huggingface_hub.snapshot_download into the HF hub cache
    # layout keyed by `cache_dir`. The runtime loads the embedder with
    # SentenceTransformer(cache_folder=MODELS_DIR), which resolves weights
    # through that same HF hub cache (ST forwards cache_folder → snapshot_download
    # cache_dir under the hood), so the model is genuinely pre-baked → instant
    # load, with no ~2GB runtime re-download that would blow the health-check
    # window on a fresh deploy.
    from huggingface_hub import snapshot_download

    print(f"Downloading embedding model: {model_name}")
    snapshot_download(
        repo_id=model_name,
        cache_dir=output_dir,
    )
    print("Embedding model downloaded.")


def download_reranker(
    output_dir: str, model_name: str = "BAAI/bge-reranker-v2-m3"
) -> None:
    # huggingface_hub.snapshot_download pulls weights + config + tokenizer
    # without triggering FlagEmbedding's compute_score path, which in some
    # FlagEmbedding↔transformers version combos hits
    # `XLMRobertaTokenizer has no attribute prepare_for_model`. The
    # `/ready` endpoint exercises the real load+inference path at runtime,
    # so we don't need an inference probe during the build.
    from huggingface_hub import snapshot_download

    print(f"Downloading reranker model: {model_name}")
    snapshot_download(
        repo_id=model_name,
        cache_dir=output_dir,
    )
    print("Reranker model downloaded.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Enrichment-Service ML models")
    parser.add_argument("--output", default="./models", help="Output directory for models")
    parser.add_argument(
        "--embedding-model", default="Qwen/Qwen3-Embedding-0.6B", help="Embedding model"
    )
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
        help="Cross-encoder reranker model",
    )
    parser.add_argument(
        "--role",
        choices=("api", "reranker", "all"),
        default="all",
        help="Cache only the model artifact needed by this runtime role",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.role in ("api", "all"):
        download_embedder(args.output, args.embedding_model)
    if args.role in ("reranker", "all"):
        download_reranker(args.output, args.reranker_model)

    print(f"\nEnrichment-Service {args.role} model artifacts downloaded to {args.output}")


if __name__ == "__main__":
    main()
