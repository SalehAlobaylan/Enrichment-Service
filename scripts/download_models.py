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
    from sentence_transformers import SentenceTransformer

    print(f"Downloading embedding model: {model_name}")
    model = SentenceTransformer(model_name, cache_folder=output_dir)
    # Probe with an Arabic + English mix — the whole point of BGE-M3 is
    # multilingual support, so verify the tokenizer handles non-Latin scripts
    # before declaring the download complete.
    test = model.encode(["test مرحبا"])
    print(f"Embedding model downloaded. Dimensions: {len(test[0])}")


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
