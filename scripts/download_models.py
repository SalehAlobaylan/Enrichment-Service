"""Pre-download Enrichment-Service ML models for Docker image build.

Whisper + CLIP live in Media-Service's download script. This now pre-caches
BAAI/bge-m3 (1024-dim multilingual text embedder). Slice A will add the
FlagEmbedding library and Slice B will add the reranker here.

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Enrichment-Service ML models")
    parser.add_argument("--output", default="./models", help="Output directory for models")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3", help="Embedding model")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    download_embedder(args.output, args.embedding_model)

    print(f"\nAll Enrichment-Service models downloaded to {args.output}")


if __name__ == "__main__":
    main()
