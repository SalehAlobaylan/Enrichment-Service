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
