from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.common.clients.cms import CMSClient
from src.common.config import Settings
from src.common.middleware.error_handler import (
    CircuitOpenError,
    EmbeddingError,
    ExtractionError,
    LLMError,
    global_error_handler,
)
from src.common.middleware.logging import LoggingMiddleware
from src.common.middleware.request_id import RequestIDMiddleware
from src.common.routes import admin, health
from src.common.utils.logging import get_logger, setup_logging
from src.extraction.routes import extract
from src.llm.clients.llm import LLMClient
from src.llm.clients.llm_cache import LLMCache
from src.llm.routes import summarize, translate
from src.retrieval.models.manager import ModelManager
from src.retrieval.routes import embed


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = Settings()
    setup_logging(log_level=settings.LOG_LEVEL, json_output=settings.is_production)
    logger = get_logger("enrichment-service")

    logger.info("starting", port=settings.PORT, env=settings.ENV)

    config_errors, config_warnings = settings.validate_startup()
    for warn in config_warnings:
        logger.warning("config_warning", error=warn)
    if config_errors:
        for err in config_errors:
            logger.error("config_invalid", error=err)
        raise RuntimeError(
            "Refusing to start: invalid configuration — "
            + "; ".join(config_errors)
        )

    model_manager = ModelManager(settings)
    cms_client = CMSClient(settings)

    # Redis-backed LLM response cache. A flaky/missing Redis must NOT block
    # boot — the cache layer treats every Redis exception as a miss.
    llm_cache: LLMCache | None = None
    redis_llm = None
    if settings.LLM_CACHE_ENABLED:
        try:
            from redis.asyncio import Redis

            redis_llm = Redis.from_url(
                settings.REDIS_URL,
                db=settings.LLM_CACHE_DB,
                decode_responses=False,
            )
            await redis_llm.ping()
            llm_cache = LLMCache(redis_llm, default_ttl_sec=settings.LLM_CACHE_TTL_SEC)
            logger.info(
                "llm_cache_ready",
                redis_url=settings.REDIS_URL,
                db=settings.LLM_CACHE_DB,
            )
        except Exception as exc:
            logger.warning(
                "llm_cache_disabled",
                reason=str(exc),
                hint="Verify REDIS_URL; cache is a perf win, not required for correctness",
            )

    llm_client = LLMClient(settings, cache=llm_cache)

    await model_manager.warmup()

    app.state.settings = settings
    app.state.model_manager = model_manager
    app.state.cms_client = cms_client
    app.state.llm_client = llm_client
    app.state.redis_llm = redis_llm

    logger.info("ready", models=model_manager.is_ready)
    yield

    await cms_client.close()
    if redis_llm is not None:
        try:
            await redis_llm.aclose()
        except Exception:
            pass
    logger.info("shutdown_complete")


app = FastAPI(
    title="Enrichment Service",
    description="Text-intelligence + retrieval microservice for the Wahb "
    "platform. Owns text embeddings, LLM-backed ops (translate, summarize, "
    "tag extraction), and Scrapling web extraction. Whisper transcription "
    "and CLIP image embedding moved to Media-Service.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — read from env at import time. Defaults are dev-friendly; production
# operators must set CORS_ALLOWED_ORIGINS explicitly (e.g. to the Platform
# Console + Wahb-Platform origins) or to "" to disable.
_cors_setting = Settings()
_cors_origins = [
    o.strip() for o in (_cors_setting.CORS_ALLOWED_ORIGINS or "").split(",") if o.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        allow_credentials=False,
    )

# Middleware (order matters — outermost first)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)

# Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Routes — transcribe + embed_image have moved to Media-Service.
app.include_router(health.router)
app.include_router(embed.router, prefix="/v1")
app.include_router(extract.router, prefix="/v1")
app.include_router(translate.router, prefix="/v1")
app.include_router(summarize.router, prefix="/v1")
app.include_router(admin.router, prefix="/v1")

# Error handlers — TranscriptionError removed (it lives in Media-Service now).
for exc_class in (CircuitOpenError, ExtractionError, EmbeddingError, LLMError):
    app.add_exception_handler(exc_class, global_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, global_error_handler)  # type: ignore[arg-type]
