# ─── Stage 1: Download ML models ──────────────────────────────
#
# Pre-caches BGE-M3 (text embedder) + bge-reranker-v2-m3 into /models so
# the runtime stage doesn't re-download ~6 GB on every container start.
# Uses PyTorch's CPU-only index — without that pin the model-downloader
# also pulls ~3 GB of nvidia-* CUDA libraries just to cache models on
# disk. FlagEmbedding is needed because download_models.py uses
# FlagReranker for the reranker download.
FROM python:3.11-slim AS model-downloader

# gcc + python headers are needed because FlagEmbedding pulls `zlib-state`
# (via the peft → accelerate → transformers chain) which has no pre-built
# wheel and compiles from source. This stage is discarded after `/models`
# is copied to the runtime stage, so the build-tools bloat doesn't ship.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch \
    sentence-transformers \
    FlagEmbedding

COPY scripts/download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py --output /models


# ─── Stage 2: Runtime ─────────────────────────────────────────
#
# Slim Debian + Python 3.11. System deps narrowed to what production
# actually uses: ffmpeg (HTML media references), libgomp1 (BLAS), curl
# + ca-certificates (healthcheck + outbound HTTPS).
#
# Chromium runtime deps (libnss3, libnspr4, libatk*, libcups2, libdrm2,
# libxkbcommon0, libxcomposite1, libxdamage1, libxrandr2, libgbm1,
# libpango-1.0-0, libcairo2, libasound2, libxshmfence1) were removed in
# Phase 8 because Playwright moved to requirements-dev.txt. The /v1/extract
# route falls back to Scrapling's plain-HTTP path in production.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=model-downloader /models /app/models

COPY requirements.txt .
# Install build tools, install Python deps (FlagEmbedding's transitive
# zlib-state needs gcc), then remove the build tools in the same layer so
# they don't ship in the final image. Saves ~250 MB vs leaving gcc
# installed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y gcc python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY src/ src/
COPY scripts/ scripts/

ENV MODELS_DIR=/app/models \
    PORT=5050

EXPOSE 5050

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD sh scripts/healthcheck.sh

# JSON array form — avoids signal-handling surprises with `sh -c` strings.
CMD ["sh", "-c", "python -m uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
