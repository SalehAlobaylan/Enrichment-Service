"""Pre-download Enrichment-Service ML models for Docker image build.

Whisper + CLIP have moved to Media-Service's download script. This now
only pre-caches the text embedder. Slice 0 will add the BGE-M3 model
and the reranker here.

Usage:
    python scripts/download_models.py [--output /path/to/models]
"""

import argparse
import os


def download_embedder(output_dir: str, model_name: str = "all-MiniLM-L6-v2") -> None:
    from sentence_transformers import SentenceTransformer

    print(f"Downloading embedding model: {model_name}")
    model = SentenceTransformer(model_name, cache_folder=output_dir)
    test = model.encode(["test"])
    print(f"Embedding model downloaded. Dimensions: {len(test[0])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Enrichment-Service ML models")
    parser.add_argument("--output", default="./models", help="Output directory for models")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    download_embedder(args.output, args.embedding_model)

    print(f"\nAll Enrichment-Service models downloaded to {args.output}")


if __name__ == "__main__":
    main()
