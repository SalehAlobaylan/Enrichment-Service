# CLAUDE.md

> Instructions for Claude Code (AI agent). Direct directives only.

---
> ## ABSOLUTE RULE
> **NEVER add AI agents as co-authors in git commits.**
> Do NOT include `Co-Authored-By: Claude` or any AI attribution — ever.
---

## What This Is

Python microservice for the Wahb platform. Owns text intelligence + future
retrieval brain:
- Text embeddings (all-MiniLM-L6-v2 today, BGE-M3 after Slice 0)
- LLM-backed text ops (translate, summarize, tag extraction)
- Stealth web extraction (Scrapling + Playwright)
- Future: reranker, retrieval orchestration (`/v1/related`, `/v1/feed/*`)

Whisper transcription and CLIP image embedding moved to **Media-Service**
(port 5051). If you find yourself wanting to add audio/image processing
here, that work belongs in Media-Service instead.

## Architecture

Triangle model — Aggregation, Enrichment, and CMS communicate directly:
- **Aggregation → Enrichment**: text embeddings (with optional tag extraction)
- **CMS → Enrichment**: on-demand text intel (translate, summarize, future related/feed)
- **Enrichment → CMS**: write-back via `/internal/*` API

## Running

```bash
# Development
cp .env.example .env
make install-dev
make dev

# Docker
make docker-up

# Tests
make test
```

## Dev Commands

| Command | Purpose |
|---------|---------|
| `make dev` | Start dev server (port 5050) |
| `make test` | Run all tests |
| `make test-unit` | Unit tests only |
| `make test-integration` | Integration tests only |
| `make lint` | Ruff linter |
| `make format` | Ruff formatter |
| `make docker-build` | Build Docker image |
| `make download-models` | Pre-download ML models |

## Key Patterns

**content_id write-back**: Every AI endpoint accepts an optional `content_id`. If present, Enrichment writes results directly to CMS via internal API and surfaces the outcome via `write_back_status` / `write_back_error`. If absent, stateless tool mode.

**Auth**: Bearer token (`SERVICE_AUTH_TOKEN`) on all `/v1/*` routes. Health/ready endpoints are unauthenticated. Single shared token works for both Enrichment and Media in dev.

**Circuit breaker**: CMS client uses CLOSED→OPEN→HALF_OPEN state machine (5 failures → open, 30s reset).

**LLM client**: Multi-provider with retry, fallback chain (`LLM_FALLBACK_PROVIDERS`), and content-addressable response cache in Redis db=1.

## Service Boundaries

| Enrichment Owns | Enrichment Cannot |
|---|---|
| Text embedding generation | Transcription (Media-Service owns Whisper) |
| Stealth web extraction (Scrapling) | Image embedding (Media-Service owns CLIP) |
| Translation / Summarization / Tag extraction (LLM) | Orchestrate pipelines or manage BullMQ queues |
| Future retrieval orchestration (`/v1/related`, `/v1/feed/*`) | Serve user-facing APIs |
| CMS write-back of text embeddings + LLM metadata | FFmpeg transcoding or media downloads |

## API Endpoints

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/health` | GET | No | Liveness check |
| `/ready` | GET | No | Readiness (embedder loaded + CMS reachable) |
| `/metrics` | GET | No | Prometheus metrics (`enrichment_*` prefix) |
| `/v1/models` | GET | Yes | Loaded model info (embedder) |
| `/v1/embed` | POST | Yes | Text(s) → vectors (with optional tag extraction) |
| `/v1/embed/query` | POST | Yes | Single text → vector (no write-back) |
| `/v1/extract` | POST | Yes | URL → clean article text |
| `/v1/translate` | POST | Yes | Text → target language |
| `/v1/summarize` | POST | Yes | Text → summary + key points |

**Moved to Media-Service** (port 5051):
- `POST /v1/transcribe` (sync)
- `POST /v1/transcribe/jobs` + `GET /v1/transcribe/jobs/:id` (async)
- `POST /v1/embed/image`

## Redis usage

- db=0 → Aggregation BullMQ (not ours)
- db=1 → **Enrichment LLM response cache** (this service)
- db=2 → Media arq job queue (not ours)
