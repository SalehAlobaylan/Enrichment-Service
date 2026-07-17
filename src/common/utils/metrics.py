"""Prometheus metrics for Enrichment-Service.

Whisper transcription + CLIP image embedding counters moved to
Media-Service. Keeping the `enrichment_*` prefix here so historical
dashboards continue to work for embed/translate/summarize/extract.
"""
from prometheus_client import Counter, Gauge, Histogram

# ─── Text embedding ─────────────────────────────────────────

embeddings_total = Counter(
    "enrichment_embeddings_total",
    "Total embedding requests",
    ["status"],
)

embedding_duration = Histogram(
    "enrichment_embedding_duration_seconds",
    "Embedding generation time",
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5],
)

# ─── Web extraction ─────────────────────────────────────────

extractions_total = Counter(
    "enrichment_extractions_total",
    "Total extraction requests",
    ["status"],
)

extraction_duration = Histogram(
    "enrichment_extraction_duration_seconds",
    "Web extraction time",
    buckets=[1, 5, 10, 15, 30],
)

# ─── LLM-backed operations ──────────────────────────────────

translations_total = Counter(
    "enrichment_translations_total",
    "Total translation requests",
    ["status"],
)

summarizations_total = Counter(
    "enrichment_summarizations_total",
    "Total summarization requests",
    ["status"],
)

llm_requests_total = Counter(
    "enrichment_llm_requests_total",
    "LLM API request outcomes (status: success|failure). One increment per request.",
    ["provider", "operation", "status"],
)

llm_retries_total = Counter(
    "enrichment_llm_retries_total",
    "LLM retry attempts (separate from request outcome counter). One increment per retry.",
    ["provider", "operation"],
)

llm_attempts_total = Counter(
    "enrichment_llm_attempts_total",
    "Individual provider attempts (status: success|failure|timeout).",
    ["provider", "operation", "status"],
)

llm_deadline_exhaustions_total = Counter(
    "enrichment_llm_deadline_exhaustions_total",
    "LLM operations stopped because their end-to-end request budget expired.",
    ["operation"],
)

llm_fallback_invocations_total = Counter(
    "enrichment_llm_fallback_invocations_total",
    "LLM fallback invocations — when the primary provider failed and a "
    "secondary provider was tried.",
    ["from_provider", "to_provider", "operation"],
)

llm_cache_hits_total = Counter(
    "enrichment_llm_cache_hits_total",
    "LLM response-cache hits (a redis hit bypassed the upstream call).",
    ["provider", "operation"],
)

llm_cache_misses_total = Counter(
    "enrichment_llm_cache_misses_total",
    "LLM response-cache misses (upstream provider was called).",
    ["provider", "operation"],
)

tag_extractions_total = Counter(
    "enrichment_tag_extractions_total",
    "Topic tag extraction outcomes "
    "(status: success | skipped_short | llm_failed | parse_failed).",
    ["status"],
)

# ─── Slice A — dense retrieval (/v1/related) ────────────────

related_requests_total = Counter(
    "enrichment_related_requests_total",
    "POST /v1/related request outcomes",
    ["status"],
)

related_duration = Histogram(
    "enrichment_related_duration_seconds",
    "POST /v1/related total time (resolve query + dense kNN)",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5],
)

# ─── Slice B — reranker + News-feed slide assembly ──────────

rerank_duration = Histogram(
    "enrichment_rerank_duration_seconds",
    "Cross-encoder rerank inference time over the dense candidate set",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5],
)

rerank_requests_total = Counter(
    "enrichment_rerank_requests_total",
    "Rerank stage outcomes (success | failure — failure falls back to dense kNN order)",
    ["status"],
)

feed_news_requests_total = Counter(
    "enrichment_feed_news_requests_total",
    "POST /v1/feed/news/slide outcomes",
    ["status"],
)

feed_news_compatibility_requests_total = Counter(
    "enrichment_feed_news_compatibility_requests_total",
    "Calls to the deprecated internal /v1/feed/news/slide compatibility helper.",
)

feed_news_duration = Histogram(
    "enrichment_feed_news_duration_seconds",
    "POST /v1/feed/news/slide total time (anchor + related + rerank + rules)",
    buckets=[0.1, 0.2, 0.5, 1, 2, 5, 10],
)

feed_slide_cache_hits_total = Counter(
    "enrichment_feed_slide_cache_hits_total",
    "News-feed slide cache hits (served from Redis, pipeline skipped).",
)

feed_slide_cache_misses_total = Counter(
    "enrichment_feed_slide_cache_misses_total",
    "News-feed slide cache misses (full retrieve+rerank pipeline ran).",
)

ranking_rules_dropped_total = Counter(
    "enrichment_ranking_rules_dropped_total",
    "Items dropped by each ranking rule "
    "(rule: freshness | source_diversity | format_quotas)",
    ["rule"],
)

llm_request_duration = Histogram(
    "enrichment_llm_request_duration_seconds",
    "LLM API request duration",
    ["provider", "operation"],
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60],
)

llm_errors_total = Counter(
    "enrichment_llm_errors_total",
    "LLM API errors by provider and error class",
    ["provider", "error_type"],
)

llm_output_invalid_total = Counter(
    "enrichment_llm_output_invalid_total",
    "LLM outputs rejected by a strict operation-specific contract.",
    ["operation", "reason"],
)

workload_admission_total = Counter(
    "enrichment_workload_admission_total",
    "Expensive-work admission outcomes.",
    ["workload", "outcome"],
)

workload_in_flight = Gauge(
    "enrichment_workload_in_flight",
    "Currently admitted expensive workloads.",
    ["workload"],
)

ai_spend_delivery_total = Counter(
    "enrichment_ai_spend_delivery_total",
    "AI spend delivery outcomes.",
    ["outcome"],
)

ai_spend_queue_depth = Gauge(
    "enrichment_ai_spend_queue_depth",
    "Queued AI spend events awaiting delivery.",
)

ai_spend_queue_oldest_age_seconds = Gauge(
    "enrichment_ai_spend_queue_oldest_age_seconds",
    "Age of the oldest delivered AI spend event.",
)

ai_spend_delivery_retries_total = Counter(
    "enrichment_ai_spend_delivery_retries_total",
    "AI spend delivery retries.",
)

# ─── CMS write-back + circuit breaker ───────────────────────

cms_writeback_total = Counter(
    "enrichment_cms_writeback_total",
    "CMS write-back attempts",
    ["endpoint", "status"],
)

circuit_state = Gauge(
    "enrichment_circuit_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["breaker"],
)

model_loaded = Gauge(
    "enrichment_model_loaded",
    "Whether a model is loaded (1=yes, 0=no)",
    ["model_name"],
)
