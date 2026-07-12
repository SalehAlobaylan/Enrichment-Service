# Enrichment-Service

The text-intelligence and retrieval brain of the Wahb platform. Owns text embeddings, LLM-backed ops (translate, summarize, topic-label, chapter generation), the cross-encoder reranker, hybrid-retrieval orchestration, News-feed slide assembly, and stealth web/feed extraction. **Aggregation** and **CMS** call it over HTTP; every AI endpoint can write its result back to CMS when a `content_id` is supplied, or run statelessly as a tool.

It does **not** transcribe audio, embed images, run FFmpeg, orchestrate the pipeline, or serve user-facing traffic — Whisper + CLIP belong to [Media-Service](../Media-Service).

**Port:** 5050 · **Deployment:** internal-only (Cranl) · **Stack:** Python 3.11+, FastAPI, sentence-transformers, Scrapling

> Full feature, architecture, and endpoint reference: [`../docs/enrichment-service.md`](../docs/enrichment-service.md). Product intent: [`../docs/PRD.md`](../docs/PRD.md).

## What it does

- **Text embedding** (`/v1/embed`, `/v1/embed/query`) — `Qwen/Qwen3-Embedding-0.6B`, 1024-dim dense (multilingual, strong Arabic) for pgvector search + story clustering.
- **LLM text ops** (`/v1/translate`, `/v1/summarize`, `/v1/topics/label`, `/v1/chapters/generate`) — multi-provider with a Gemini → DeepSeek default chain and a Redis response cache.
- **Reranking** (`/v1/rerank`) — `BAAI/bge-reranker-v2-m3` cross-encoder.
- **Retrieval** (`/v1/related`, `/v1/feed/news/slide`) — hybrid kNN + RRF fusion and News-feed slide assembly.
- **Extraction** (`/v1/extract`, `/extract/{feed,telegram,twitter}`) — Scrapling + Playwright stealth.

## Media Atomization Role

Enrichment plans contextual chapters for long media through `/v1/chapters/generate`; it does not cut media or publish feed units. The chapter plan should include title, summary, start/end timestamps, context label, confidence, boundary reason, standalone/coherence score, sponsor/intro/outro indicators, and review reason when needed. Aggregation performs deterministic duration normalization and merging after the plan; CMS owns review state and feed visibility.

## Run Locally

```bash
cp .env.example .env
make dev          # creates .venv, installs deps + Playwright Chromium, starts API on :5050
```

- **Health:** `GET /health`, `GET /ready` — no auth
- **API:** `/v1/*` — requires `Authorization: Bearer <SERVICE_AUTH_TOKEN>`
- **Metrics:** `GET /metrics` — Prometheus (`enrichment_*`)

### Docker

```bash
make docker-up        # or: make docker-build
```

## Configuration

Env is for boot-time infrastructure only. Retrieval **tuning knobs** (`RRF_K`, `RELATED_K_*`, `RERANK_INPUT_K`, `FRESHNESS_DECAY_*`, `NEWS_MAX_PER_SOURCE`) are code defaults, deliberately kept out of `.env.example` — env override is an emergency escape hatch (Config Discipline). See `.env.example` for the full list.

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `SERVICE_AUTH_TOKEN` | prod | — | Bearer token for `/v1/*` (falls back to `ENRICHMENT_SERVICE_TOKEN` / `CMS_SERVICE_TOKEN`) |
| `CMS_BASE_URL` | yes | http://localhost:8080 | CMS for `content_id` write-back |
| `CMS_SERVICE_TOKEN` | yes | — | Token for CMS `/internal/*` writes |
| `PORT` | no | 5050 | HTTP port |
| `ENV` | no | development | `production` makes missing keys/token fatal |
| `EMBEDDING_MODEL` | no | `Qwen/Qwen3-Embedding-0.6B` | Text embedder selector |
| `EMBEDDING_MODEL_REVISION` | no | empty | Immutable model commit; empty auto-resolves the loaded snapshot for lifecycle provenance |
| `RERANKER_MODEL` | no | `BAAI/bge-reranker-v2-m3` | Cross-encoder selector |
| `RERANK_ENABLED` | no | true | Reranker toggle |
| `MODELS_DIR` | no | ./models | Model cache dir |
| `LLM_PROVIDER` | no | gemini | Primary LLM provider |
| `LLM_MODEL` | no | gemini-3.5-flash | Primary model |
| `LLM_FALLBACK_PROVIDERS` | no | deepseek | CSV fallback chain |
| `GEMINI_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | per provider | — | Provider keys (missing key is fatal in prod) |
| `LLM_CACHE_ENABLED` / `LLM_CACHE_TTL_SEC` | no | true / 604800 | LLM response cache (7-day TTL) |
| `FEED_SLIDE_CACHE_ENABLED` / `FEED_SLIDE_CACHE_TTL_SEC` | no | true / 300 | News-slide read cache |
| `REDIS_URL` / `LLM_CACHE_DB` | no | redis://localhost:6379 / 1 | LLM + slide cache Redis |
| `ENRICHMENT_ROLE` | no | api | `api` or `reranker` (split deploy — see below) |
| `RERANKER_BASE_URL` | no | — | Remote reranker URL when `role=api` is split |
| `EXTRACT_TIMEOUT_SEC` | no | 30 | Scrapling timeout |
| `CB_FAILURE_THRESHOLD` / `CB_RESET_TIMEOUT_SEC` / `CB_HALF_OPEN_REQUESTS` | no | 5 / 30 / 3 | CMS circuit breaker |

**Referenced in code but not in `.env.example`:** `CORS_ALLOWED_ORIGINS` (CSV; "" disables CORS), and `TWITTER_GQL_USER_BY_SCREENNAME` / `TWITTER_GQL_USER_TWEETS` (GraphQL endpoint IDs for `/v1/extract/twitter`).

## Patterns

- **`content_id` write-back** — every AI endpoint accepts an optional `content_id`; if present, results are written to CMS via `/internal` and the outcome is surfaced via `write_back_status` / `write_back_error`. Absent → stateless tool mode.
- **Reranker split deploy** — one Cranl instance can't hold both the embedder and the reranker in fixed RAM. `ENRICHMENT_ROLE=reranker` loads only the cross-encoder and serves only `/v1/rerank`; `ENRICHMENT_ROLE=api` + `RERANKER_BASE_URL` loads only the embedder and calls the remote reranker over HTTP; `api` with no `RERANKER_BASE_URL` is the local monolith (loads both).
- **Redis db conventions** — db=0 Aggregation BullMQ · **db=1 Enrichment LLM + slide cache (this service)** · db=2 Media arq queue.

## Commands

| Command | Purpose |
|---------|---------|
| `make dev` | Dev server with reload (:5050) |
| `make run` | Production-style local run |
| `make install` / `make install-dev` | Install runtime / dev deps into `.venv` |
| `make test` / `make test-unit` / `make test-integration` | Tests |
| `make test-coverage` | Coverage report |
| `make lint` / `make format` | Ruff |
| `make download-models` | Pre-download embedder + reranker |

## Project Structure

```
src/
├── main.py            # app + role-based router mounting, middleware, metrics
├── common/            # config, CMS client, middleware, health/admin routes
├── retrieval/         # embed, related, feed_news, rerank + sentence-transformers ModelManager
├── llm/               # translate, summarize, topic_label, chapters + LLMClient + Redis cache
└── extraction/        # Scrapling routes (URL / feed / telegram / twitter)
tests/                 # unit + integration
scripts/               # model download, ops helpers
```
