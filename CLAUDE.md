# CLAUDE.md

> Instructions for Claude Code (AI agent). Direct directives only.

---
> ## ABSOLUTE RULE
> **NEVER add AI agents as co-authors in git commits.**
> Do NOT include `Co-Authored-By: Claude` or any AI attribution — ever.
---

## What This Is

Python microservice for the Wahb platform. Handles all AI/ML and advanced scraping:
- Whisper transcription (faster-whisper)
- Sentence embeddings (all-MiniLM-L6-v2, 384-dim)
- Web extraction (Scrapling + Playwright stealth)
- Translation (LLM API)
- Summarization (LLM API)

## Architecture

Triangle model — Aggregation, Enrichment, and CMS communicate directly:
- **Aggregation → Enrichment**: pipeline AI tasks (transcribe, embed, extract)
- **CMS → Enrichment**: on-demand AI (translate, summarize, embed query)
- **Enrichment → CMS**: write-back results via `/internal/*` API

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

**content_id write-back**: Every AI endpoint accepts an optional `content_id`. If present, Enrichment writes results directly to CMS via internal API. If absent, stateless tool mode.

**Auth**: Bearer token (`SERVICE_AUTH_TOKEN`) on all `/v1/*` routes. Health/ready endpoints are unauthenticated.

**Circuit breaker**: CMS client uses CLOSED→OPEN→HALF_OPEN state machine (5 failures → open, 30s reset).

## Service Boundaries

| Enrichment Owns | Enrichment Cannot |
|---|---|
| Whisper transcription | Orchestrate pipelines |
| Stealth web extraction | Manage BullMQ queues |
| Embeddings generation | Serve user-facing APIs |
| Translation (LLM) | Write content items (only enrichment fields) |
| Summarization (LLM) | CMS business logic |

## API Endpoints

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/health` | GET | No | Liveness check |
| `/ready` | GET | No | Readiness (models loaded + CMS reachable) |
| `/metrics` | GET | No | Prometheus metrics |
| `/v1/models` | GET | Yes | Loaded model info |
| `/v1/transcribe` | POST | Yes | Audio/video → text |
| `/v1/embed` | POST | Yes | Text(s) → 384-dim vectors |
| `/v1/embed/query` | POST | Yes | Single text → vector (no write-back) |
| `/v1/extract` | POST | Yes | URL → clean article text |
| `/v1/translate` | POST | Yes | Text → target language |
| `/v1/summarize` | POST | Yes | Text → summary + key points |
