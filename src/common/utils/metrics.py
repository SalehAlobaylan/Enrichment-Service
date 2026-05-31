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

# ─── Slice A — hybrid retrieval (/v1/related) ───────────────

related_requests_total = Counter(
    "enrichment_related_requests_total",
    "POST /v1/related request outcomes",
    ["status"],
)

related_duration = Histogram(
    "enrichment_related_duration_seconds",
    "POST /v1/related total time (resolve query + 2× kNN + RRF fusion)",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5],
)

rrf_fusion_overlap_ratio = Gauge(
    "enrichment_rrf_fusion_overlap_ratio",
    "Fraction of fused results that came from both dense + sparse rankings. "
    "High = the two modes are redundant for this corpus; low = hybrid is "
    "pulling its weight by surfacing items that pure dense or sparse miss.",
)

# ─── Slice B — reranker + News-feed slide assembly ──────────

rerank_duration = Histogram(
    "enrichment_rerank_duration_seconds",
    "Cross-encoder rerank inference time over the post-RRF candidate set",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5],
)

rerank_requests_total = Counter(
    "enrichment_rerank_requests_total",
    "Rerank stage outcomes (success | failure — failure falls back to RRF order)",
    ["status"],
)

feed_news_requests_total = Counter(
    "enrichment_feed_news_requests_total",
    "POST /v1/feed/news/slide outcomes",
    ["status"],
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
    "(rule: freshness | source_diversity | type_quotas)",
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

# ─── CMS write-back + circuit breaker ───────────────────────

cms_writeback_total = Counter(
    "enrichment_cms_writeback_total",
    "CMS write-back attempts",
    ["endpoint", "status"],
)

circuit_state = Gauge(
    "enrichment_circuit_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
)

model_loaded = Gauge(
    "enrichment_model_loaded",
    "Whether a model is loaded (1=yes, 0=no)",
    ["model_name"],
)
