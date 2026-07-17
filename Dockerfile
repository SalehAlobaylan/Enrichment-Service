# `pyproject.toml` + `uv.lock` are the only dependency authority. Every stage
# installs the same frozen CPU-Torch graph; model stages differ only in the
# artifact they cache.
FROM ghcr.io/astral-sh/uv:0.9.28 AS uv

FROM python:3.11-slim AS runtime-deps
COPY --from=uv /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml uv.lock ./

# FlagEmbedding may build zlib-state on platforms without a wheel. Keep the
# toolchain only for this layer, then purge it before any runtime target.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libgomp1 \
        ca-certificates \
    && uv sync --frozen --no-dev --no-install-project \
    && apt-get purge -y gcc python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

FROM runtime-deps AS model-api
COPY scripts/download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py --output /models --role api

FROM runtime-deps AS model-reranker
COPY scripts/download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py --output /models --role reranker

FROM runtime-deps AS runtime-base
WORKDIR /app
RUN groupadd --system enrichment \
    && useradd --system --gid enrichment --home /app enrichment \
    && mkdir /app/models \
    && chown -R enrichment:enrichment /app /opt/venv

COPY --chown=enrichment:enrichment src/ src/
COPY --chown=enrichment:enrichment scripts/ scripts/

USER enrichment
ENV MODELS_DIR=/app/models \
    PORT=5050
EXPOSE 5050
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD sh scripts/healthcheck.sh

FROM runtime-base AS api
COPY --chown=enrichment:enrichment --from=model-api /models /app/models
ENV ENRICHMENT_ROLE=api
CMD ["sh", "-c", "python -m uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]

FROM runtime-base AS reranker
COPY --chown=enrichment:enrichment --from=model-reranker /models /app/models
ENV ENRICHMENT_ROLE=reranker
CMD ["sh", "-c", "python -m uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
