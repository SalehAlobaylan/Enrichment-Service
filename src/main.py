from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src.clients.cms import CMSClient
from src.clients.llm import LLMClient
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
from src.routes import admin, embed, extract, health, summarize, transcribe, translate
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
    llm_client = LLMClient(settings)

    await model_manager.warmup()

    app.state.settings = settings
    app.state.model_manager = model_manager
    app.state.cms_client = cms_client
    app.state.llm_client = llm_client

    logger.info("ready", models=model_manager.is_ready)
    yield

    await cms_client.close()
    logger.info("shutdown_complete")


app = FastAPI(
    title="Enrichment Service",
    description="AI/ML enrichment microservice for the Wahb platform",
    version="1.0.0",
    lifespan=lifespan,
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
app.include_router(extract.router, prefix="/v1")
app.include_router(translate.router, prefix="/v1")
app.include_router(summarize.router, prefix="/v1")
app.include_router(admin.router, prefix="/v1")

# Error handlers
for exc_class in (CircuitOpenError, TranscriptionError, ExtractionError, EmbeddingError, LLMError):
    app.add_exception_handler(exc_class, global_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, global_error_handler)  # type: ignore[arg-type]
