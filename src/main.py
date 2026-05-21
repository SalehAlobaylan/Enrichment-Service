from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.clients.cms import CMSClient
from src.clients.llm import LLMClient
from src.clients.llm_cache import LLMCache
from src.config import Settings
from src.middleware.error_handler import (
    CircuitOpenError,
    EmbeddingError,
    ExtractionError,
    LLMError,
    TranscriptionError,
    global_error_handler,
)
from src.middleware.logging import LoggingMiddleware
from src.middleware.request_id import RequestIDMiddleware
from src.models.manager import ModelManager
from src.routes import (
    admin,
    embed,
    embed_image,
    extract,
    health,
    summarize,
    transcribe,
    translate,
)
from src.utils.logging import get_logger, setup_logging


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

    # arq pool for enqueueing async transcription jobs (#1). Same Redis,
    # different logical DB. The API only enqueues; a separate `arq` worker
    # process executes the jobs.
    arq_pool = None
    try:
        from arq import create_pool

        from src.worker import _build_redis_settings

        arq_pool = await create_pool(_build_redis_settings())
        logger.info("arq_pool_ready", db=settings.ARQ_REDIS_DB)
    except Exception as exc:
        logger.warning(
            "arq_pool_disabled",
            reason=str(exc),
            hint="Async transcription endpoints will return 503 until Redis is reachable",
        )

    await model_manager.warmup()

    app.state.settings = settings
    app.state.model_manager = model_manager
    app.state.cms_client = cms_client
    app.state.llm_client = llm_client
    app.state.redis_llm = redis_llm
    app.state.arq_pool = arq_pool

    logger.info("ready", models=model_manager.is_ready)
    yield

    await cms_client.close()
    if redis_llm is not None:
        try:
            await redis_llm.aclose()
        except Exception:
            pass
    if arq_pool is not None:
        try:
            await arq_pool.aclose()
        except Exception:
            pass
    logger.info("shutdown_complete")


app = FastAPI(
    title="Enrichment Service",
    description="AI/ML enrichment microservice for the Wahb platform",
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

# Routes
app.include_router(health.router)
app.include_router(transcribe.router, prefix="/v1")
app.include_router(embed.router, prefix="/v1")
app.include_router(embed_image.router, prefix="/v1")
app.include_router(extract.router, prefix="/v1")
app.include_router(translate.router, prefix="/v1")
app.include_router(summarize.router, prefix="/v1")
app.include_router(admin.router, prefix="/v1")

# Error handlers
for exc_class in (CircuitOpenError, TranscriptionError, ExtractionError, EmbeddingError, LLMError):
    app.add_exception_handler(exc_class, global_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, global_error_handler)  # type: ignore[arg-type]
