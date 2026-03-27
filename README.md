# Enrichment-Service

Python service for transcription (Whisper), text embeddings, URL extraction, and LLM translation/summarization. **Aggregation** and **CMS** call it over HTTP; it can write results back to CMS when a `content_id` is supplied.

## Run locally

```bash
cp .env.example .env
make dev
```

`make dev` creates `.venv` on first run, installs dependencies, installs Playwright Chromium, and starts the API on `http://localhost:5050`.

- **Health:** `GET /health` — no auth  
- **API:** `GET/POST /v1/...` — `Authorization: Bearer <SERVICE_AUTH_TOKEN>`

## Environment (minimum)

| Variable | Purpose |
|----------|---------|
| `SERVICE_AUTH_TOKEN` | Required on `/v1/*` |
| `CMS_BASE_URL` | CMS base URL for write-back |
| `CMS_SERVICE_TOKEN` | Token for CMS `/internal/*` calls |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Translate/summarize (per `LLM_PROVIDER`) |

Optional: `PORT` (default **5050**), `WHISPER_*`, `MODELS_DIR`, `LOG_LEVEL`. Full list: `../context/Enrichment_Service_Context_Requirements.md`.

## Docker

```bash
make docker-up
# or: make docker-build
```

Docker and health checks now use `PORT` consistently, defaulting to **5050**.

## Commands

| Command | |
|---------|---|
| `make run` | Production-style local run |
| `make dev` | Dev server with reload |
| `make install` / `make install-dev` | Install runtime or dev deps into `.venv` |
| `make test` | Tests |
| `make lint` / `make format` | Ruff |
| `make download-models` | Pre-download Whisper + embedding models |

## Contracts & details

Endpoint table, error codes, circuit breaker, and integration with Aggregation/CMS: **`context/Enrichment_Service_Context_Requirements.md`** at the repo root.
