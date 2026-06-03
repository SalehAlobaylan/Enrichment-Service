"""Pre-download Enrichment-Service ML models for Docker image build.

Whisper + CLIP live in Media-Service's download script. This pre-caches the
two text-side models hosted by Enrichment:

  - BAAI/bge-m3:                multilingual text embedder (dense + sparse)
  - BAAI/bge-reranker-v2-m3:    cross-encoder reranker (Slice B)

Usage:
    python scripts/download_models.py [--output /path/to/models]
"""

import argparse
import os


def download_embedder(output_dir: str, model_name: str = "BAAI/bge-m3") -> None:
    # CRITICAL: pre-cache via huggingface_hub.snapshot_download (HF hub cache
    # layout), NOT SentenceTransformer(cache_folder=...). The RUNTIME loads
    # BGE-M3 with FlagEmbedding's BGEM3FlagModel (see retrieval/models/
    # embedder.py), which resolves weights through the HF hub cache keyed by
    # `cache_dir`. SentenceTransformer's cache_folder uses a DIFFERENT on-disk
    # layout, so a model pre-cached that way is invisible to FlagEmbedding —
    # the container then re-downloads ~3GB ("Fetching 30 files") at startup,
    # blowing the health-check window so a fresh deploy never goes ready.
    # Mirroring download_reranker (same snapshot_download + cache_dir) makes the
    # embedder genuinely pre-baked → instant load, no runtime download.
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
    parser.add_argument("--embedding-model", default="BAAI/bge-m3", help="Embedding model")
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
        help="Cross-encoder reranker model",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    download_embedder(args.output, args.embedding_model)
    download_reranker(args.output, args.reranker_model)

    print(f"\nAll Enrichment-Service models downloaded to {args.output}")


if __name__ == "__main__":
    main()
