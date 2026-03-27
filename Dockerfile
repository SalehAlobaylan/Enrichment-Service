# ─── Stage 1: Download ML models ──────────────────────────────
FROM python:3.11-slim AS model-downloader

RUN pip install --no-cache-dir faster-whisper sentence-transformers

COPY scripts/download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py --output /models

# ─── Stage 2: Install Playwright + Chromium ───────────────────
FROM python:3.11-slim AS browser

RUN pip install --no-cache-dir playwright \
    && playwright install chromium --with-deps

# ─── Stage 3: Runtime ─────────────────────────────────────────
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    curl \
    # Chromium runtime deps
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-downloaded models
COPY --from=model-downloader /models /app/models

# Playwright Chromium
COPY --from=browser /root/.cache/ms-playwright /root/.cache/ms-playwright

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code
COPY src/ src/
COPY scripts/ scripts/

ENV MODELS_DIR=/app/models
ENV PORT=5050
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright

EXPOSE 5050

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD sh scripts/healthcheck.sh

CMD sh -c "python -m uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"
