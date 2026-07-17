# Enrichment-Service

The text-intelligence and retrieval brain of the Wahb platform. Owns text embeddings, LLM-backed ops (translate, summarize, topic-label, chapter generation), the cross-encoder reranker, dense retrieval orchestration, an internal News-ranking helper, and bounded web/feed extraction. **Aggregation** and **CMS** call it over HTTP; only embedding and translate/summarize support CMS write-back, while the remaining endpoints are stateless tools.

It does **not** transcribe audio, embed images, run FFmpeg, orchestrate the pipeline, or serve user-facing traffic — Whisper + CLIP belong to [Media-Service](../Media-Service).

**Port:** 5050 · **Deployment:** internal-only (Cranl) · **Stack:** Python 3.11+, FastAPI, sentence-transformers, Scrapling

> Full feature, architecture, and endpoint reference: [`../docs/enrichment-service.md`](../docs/enrichment-service.md). Product intent: [`../docs/PRD.md`](../docs/PRD.md).

## What it does

- **Text embedding** (`/v1/embed`, `/v1/embed/query`) — `Qwen/Qwen3-Embedding-0.6B`, 1024-dim dense (multilingual, strong Arabic) for pgvector search + story clustering.
- **LLM text ops** (`/v1/translate`, `/v1/summarize`, `/v1/topics/label`, `/v1/chapters/generate`) — multi-provider with a Gemini → DeepSeek default chain and a Redis response cache.
- **Reranking** (`/v1/rerank`) — `BAAI/bge-reranker-v2-m3` cross-encoder.
- **Retrieval** (`/v1/related`, `/v1/feed/news/slide`) — dense Qwen kNN with optional reranking and a compatibility slide helper.
- **Extraction** (`/v1/extract`, `/extract/{feed,telegram,twitter}`) — bounded curl_cffi transport + Scrapling parsing, without a browser runtime.

## Media Atomization Role

Enrichment plans contextual chapters for long media through `/v1/chapters/generate`; it does not cut media or publish feed units. The chapter plan should include title, summary, start/end timestamps, context label, confidence, boundary reason, standalone/coherence score, sponsor/intro/outro indicators, and review reason when needed. Aggregation performs deterministic duration normalization and merging after the plan; CMS owns review state and feed visibility.

## Retrieval boundary

Qwen retrieval is dense-only. `/v1/related` accepts canonical kinds (`NEWS`,
`VIDEO`, `PODCAST`) and, independently, NEWS formats (`ARTICLE`, `TWEET`,
`COMMENT`); it never calls the legacy sparse kNN surface. The
`/v1/feed/news/slide` endpoint is retained as a deprecated internal
compatibility/ranking helper with a versioned cache key. It is not a public feed
API: CMS remains the owner of story clustering, public News-feed assembly,
visibility, and serving. Its use is counted by
`enrichment_feed_news_compatibility_requests_total`; remove the route after a
30-day zero-usage observation window.

## Run Locally

```bash
cp .env.example .env
make dev          # creates .venv, installs dependencies, starts API on :5050
```

- **Health:** `GET /health`, `GET /ready` — no auth
- **API:** `/v1/*` — requires `Authorization: Bearer <SERVICE_AUTH_TOKEN>`
- **Metrics:** `GET /metrics` — Prometheus (`enrichment_*`)

## Request limits and admission

The service rejects request bodies above **1,000,000 bytes** before JSON parsing.
Semantic limits are enforced at the route schema boundary: embedding and related
text are capped at 12,000 characters, embedding batches at 32 entries, rerank
inputs at 32 candidates of 4,000 characters, source references at 512 characters,
search queries at 400 characters, and discovery fan-out at 40 results (50 pasted
YouTube references). Oversized or inconsistent batches receive a 413/422 before
model, provider, or CMS work starts.

Accepted expensive work is isolated into lifecycle-owned thread pools: embedding
(2), reranking (1), extraction (4), and synchronous provider calls (4). A short
admission wait returns retryable `429 WORKLOAD_OVERLOADED` with `Retry-After: 1`
instead of queuing unbounded work. These are code policy defaults, not extra env
tuning knobs.

### Docker

```bash
make docker-up                   # or: make docker-build
make docker-build-reranker       # build the reranker image only
```

Compose starts the split `enrichment-api` and `enrichment-reranker` roles; it
does not default to a 4 GB monolith. Docker targets `api` and `reranker` each
carry only their required model artifact and run as the `enrichment` user.

## Configuration

Env is for boot-time infrastructure only. Retrieval **tuning knobs** (`RELATED_K_DENSE_DEFAULT`, `RERANK_INPUT_K`, `FRESHNESS_DECAY_*`, `NEWS_MAX_PER_SOURCE`) are code defaults, deliberately kept out of `.env.example` — env override is an emergency escape hatch (Config Discipline). See `.env.example` for the full list.

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `SERVICE_AUTH_TOKEN` | prod | — | Bearer token for inbound `/v1/*` calls; distinct from CMS/reranker credentials in production |
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
| `RERANKER_SERVICE_TOKEN` | remote prod | — | Dedicated API→reranker bearer credential |
| `EXTRACT_TIMEOUT_SEC` | no | 30 | Scrapling timeout |
| `CB_FAILURE_THRESHOLD` / `CB_RESET_TIMEOUT_SEC` / `CB_HALF_OPEN_REQUESTS` | no | 5 / 30 / 3 | CMS circuit breaker |

**Referenced in code but not in `.env.example`:** `CORS_ALLOWED_ORIGINS` (CSV; "" disables CORS), and `TWITTER_GQL_USER_BY_SCREENNAME` / `TWITTER_GQL_USER_TWEETS` (GraphQL endpoint IDs for `/v1/extract/twitter`).

## Patterns

- **Selective write-back** — `/v1/embed` writes supplied `content_ids`; `/v1/translate` and `/v1/summarize` write an optional `content_id`, each reporting `write_back_status` / `write_back_error`. The other endpoints are stateless tools.
- **Reranker split deploy** — one Cranl instance can't hold both the embedder and the reranker in fixed RAM. `ENRICHMENT_ROLE=reranker` loads only the cross-encoder and serves only `/v1/rerank`; `ENRICHMENT_ROLE=api` + `RERANKER_BASE_URL` loads only the embedder and calls the remote reranker over HTTP; `api` with no `RERANKER_BASE_URL` is the local monolith (loads both).
- **Redis db conventions** — db=0 Aggregation BullMQ · **db=1 Enrichment LLM + slide cache (this service)** · db=2 Media arq queue.

## Commands

| Command | Purpose |
|---------|---------|
| `make dev` | Dev server with reload (:5050) |
| `make run` | Production-style local run |
| `make install` / `make install-dev` | Install runtime / dev deps into `.venv` |
| `make dependency-check` | Verify `pyproject.toml` matches the checked `uv.lock` |
| `make test` / `make test-unit` / `make test-integration` | Tests |
| `make test-coverage` | Coverage report |
| `make lint` / `make format` | Ruff |
| `make download-models` | Pre-download embedder + reranker |

`pyproject.toml` and `uv.lock` are the dependency authority for local, CI,
model-download, and Docker environments. Requirements exports are deliberately
not tracked; generate one with `make export-requirements` only for a consumer
that cannot use uv.

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
