from prometheus_client import Counter, Gauge, Histogram

transcriptions_total = Counter(
    "enrichment_transcriptions_total",
    "Total transcription requests",
    ["status", "model_size"],
)

transcription_duration = Histogram(
    "enrichment_transcription_duration_seconds",
    "Transcription processing time",
    ["model_size"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

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

transcribe_jobs_total = Counter(
    "enrichment_transcribe_jobs_total",
    "Async transcription jobs by state (queued | started | completed | failed).",
    ["state"],
)

image_embeddings_total = Counter(
    "enrichment_image_embeddings_total",
    "Total CLIP image embedding requests",
    ["status"],
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
