import asyncio
import os
import resource
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.common.clients.cms import CMSClient
from src.common.config import Settings
from src.common.middleware.body_limit import RequestBodyLimitMiddleware
from src.common.middleware.error_handler import (
    CircuitOpenError,
    EmbeddingError,
    ExtractionError,
    LLMError,
    WorkloadOverloadedError,
    global_error_handler,
)
from src.common.middleware.logging import LoggingMiddleware
from src.common.middleware.request_id import RequestIDMiddleware
from src.common.migration_control import MigrationFenceMiddleware
from src.common.migration_control import owner as migration_owner
from src.common.migration_control import router as migration_router
from src.common.routes import admin, artifact_recovery, health
from src.common.utils.logging import get_logger, setup_logging
from src.common.workload_admission import WorkloadAdmission, WorkloadExecutors
from src.extraction.routes import extract
from src.llm.clients.llm import LLMClient
from src.llm.clients.llm_cache import LLMCache
from src.llm.clients.spend_meter import SpendMeter
from src.llm.routes import (
    chapter_proposal,
    chapters,
    classify,
    summarize,
    topic_digest,
    topic_label,
    translate,
)
from src.operator.routes import reason as operator_reason
from src.retrieval.clients.slide_cache import SlideCache
from src.retrieval.models.manager import ModelManager
from src.retrieval.routes import embed, feed_news, related, rerank


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = getattr(app.state, "bootstrap_settings", None) or Settings()
    setup_logging(log_level=settings.LOG_LEVEL, json_output=settings.is_production)
    logger = get_logger("enrichment-service")

    logger.info("starting", port=settings.PORT, env=settings.ENV)

    config_errors, config_warnings = settings.validate_startup()
    for warn in config_warnings:
        logger.warning("config_warning", error=warn)
    if config_errors:
        for err in config_errors:
            logger.error("config_invalid", error=err)
        raise RuntimeError("Refusing to start: invalid configuration — " + "; ".join(config_errors))

    migration_redis = None
    try:
        from redis.asyncio import Redis

        migration_redis = Redis.from_url(
            settings.REDIS_URL, db=settings.LLM_CACHE_DB, decode_responses=False
        )
        await migration_redis.ping()
        await migration_owner.restore(migration_redis)
    except Exception as exc:
        logger.error("migration_owner_store_unavailable", reason=str(exc))
        if migration_redis is not None:
            await migration_redis.aclose()
        migration_redis = None

    model_manager = ModelManager(settings)
    if model_manager.role == "reranker":
        await model_manager.warmup()
        app.state.settings = settings
        app.state.model_manager = model_manager
        app.state.cms_client = None
        app.state.llm_client = None
        app.state.redis_llm = None
        app.state.migration_redis = migration_redis
        app.state.slide_cache = None
        app.state.workload_admission = WorkloadAdmission()
        app.state.workload_executors = WorkloadExecutors()
        logger.info("ready", models=model_manager.is_ready)
        yield
        model_manager.close()
        app.state.workload_executors.close()
        if migration_redis is not None:
            await migration_redis.aclose()
        logger.info("shutdown_complete")
        return

    cms_client = CMSClient(settings)

    # Redis-backed LLM response cache. A flaky/missing Redis must NOT block
    # boot — the cache layer treats every Redis exception as a miss.
    llm_cache: LLMCache | None = None
    redis_llm = None
    if settings.LLM_CACHE_ENABLED:
        try:
            redis_llm = migration_redis
            if redis_llm is None:
                raise RuntimeError("Redis migration owner connection is unavailable")
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

    workload_admission = WorkloadAdmission()
    workload_executors = WorkloadExecutors()
    spend_meter = SpendMeter(cms_client)
    spend_meter.start()
    llm_client = LLMClient(
        settings,
        cache=llm_cache,
        cms_client=cms_client,
        admission=workload_admission,
        executors=workload_executors,
        spend_meter=spend_meter,
    )

    # News-feed slide cache — reuses the LLM-cache Redis connection (db=1) with
    # a distinct key prefix. None if the cache is off or Redis was unreachable
    # above; FeedNewsService then always computes.
    slide_cache: SlideCache | None = None
    if settings.FEED_SLIDE_CACHE_ENABLED and redis_llm is not None:
        slide_cache = SlideCache(redis_llm, default_ttl_sec=settings.FEED_SLIDE_CACHE_TTL_SEC)
        logger.info("slide_cache_ready", ttl_sec=settings.FEED_SLIDE_CACHE_TTL_SEC)

    await model_manager.warmup()

    app.state.settings = settings
    app.state.model_manager = model_manager
    app.state.cms_client = cms_client
    app.state.llm_client = llm_client
    app.state.redis_llm = redis_llm
    app.state.migration_redis = migration_redis
    app.state.slide_cache = slide_cache
    app.state.workload_admission = workload_admission
    app.state.workload_executors = workload_executors
    app.state.spend_meter = spend_meter

    async def publish_lane_snapshot(lane: str) -> None:
        """Publish a compact, current Enrichment view to CMS.

        Aggregation owns queue depth; Enrichment owns admission and model
        saturation. Keeping both in the CMS snapshot makes a green HTTP probe
        distinguishable from a lane that is rejecting work locally.
        """
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_bytes = int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024)
        await cms_client.put_pipeline_lane_snapshot(lane, {
            "required_queue_depth": 0,
            "optional_queue_depth": 0,
            "required_oldest_age_seconds": 0,
            "optional_oldest_age_seconds": 0,
            "dlq_delta": 0,
            "enrichment_counts": workload_admission.snapshot(lane),
            "process_metrics": {"pid": os.getpid(), "max_rss_bytes": rss_bytes},
            "resource_metrics": {"embedding_capacity": workload_admission.capacity("embedding")},
        })

    async def publish_lane_snapshots() -> None:
        while True:
            try:
                for lane in ("news", "pods"):
                    await publish_lane_snapshot(lane)
            except Exception as exc:
                logger.debug("pipeline_snapshot_deferred", reason=str(exc))
            await asyncio.sleep(15)

    app.state.pipeline_snapshot_task = asyncio.create_task(publish_lane_snapshots())

    logger.info("ready", models=model_manager.is_ready)
    yield

    app.state.pipeline_snapshot_task.cancel()
    await asyncio.gather(app.state.pipeline_snapshot_task, return_exceptions=True)
    await spend_meter.close()
    await llm_client.close()
    await cms_client.close()
    model_manager.close()
    workload_executors.close()
    if migration_redis is not None:
        try:
            await migration_redis.aclose()
        except Exception:
            pass
    logger.info("shutdown_complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build one role-specific application from one immutable settings object."""
    settings = settings or Settings()
    app = FastAPI(
        title="Enrichment Service",
        description="Text-intelligence + retrieval microservice for the Wahb platform.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.bootstrap_settings = settings

    cors_origins = [
        origin.strip()
        for origin in (settings.CORS_ALLOWED_ORIGINS or "").split(",")
        if origin.strip()
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            allow_credentials=False,
        )
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(MigrationFenceMiddleware)
    app.add_middleware(RequestBodyLimitMiddleware)
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    app.include_router(health.router)
    app.include_router(migration_router)
    app.include_router(rerank.router, prefix="/v1")
    if settings.ENRICHMENT_ROLE.strip().lower() != "reranker":
        app.include_router(embed.router, prefix="/v1")
        app.include_router(related.router, prefix="/v1")
        app.include_router(feed_news.router, prefix="/v1")
        app.include_router(extract.router, prefix="/v1")
        app.include_router(translate.router, prefix="/v1")
        app.include_router(summarize.router, prefix="/v1")
        app.include_router(topic_label.router, prefix="/v1")
        app.include_router(classify.router, prefix="/v1")
        app.include_router(topic_digest.router, prefix="/v1")
        app.include_router(chapters.router, prefix="/v1")
        app.include_router(chapter_proposal.router, prefix="/v1")
        app.include_router(operator_reason.router, prefix="/v1/operator")
        app.include_router(admin.router, prefix="/v1")
        app.include_router(artifact_recovery.router, prefix="/internal/artifact-recovery")

    for exc_class in (
        CircuitOpenError,
        ExtractionError,
        EmbeddingError,
        LLMError,
        WorkloadOverloadedError,
    ):
        app.add_exception_handler(exc_class, global_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, global_error_handler)
    return app


app = create_app()
