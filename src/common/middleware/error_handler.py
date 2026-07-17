from fastapi import Request
from fastapi.responses import JSONResponse

from src.common.utils.logging import get_logger
from src.common.workload_admission import RETRY_AFTER_SEC, WorkloadOverloadedError

logger = get_logger(__name__)


class CircuitOpenError(Exception):
    pass


class TranscriptionError(Exception):
    pass


class ExtractionError(Exception):
    pass


class EmbeddingError(Exception):
    pass


class LLMError(Exception):
    pass


async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")

    if isinstance(exc, CircuitOpenError):
        return JSONResponse(
            status_code=503,
            content={
                "error": "CMS service unavailable — circuit breaker is open",
                "error_code": "CIRCUIT_OPEN",
                "retryable": True,
                "retry_after_seconds": 30,
                "request_id": request_id,
            },
        )

    if isinstance(exc, WorkloadOverloadedError):
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(RETRY_AFTER_SEC)},
            content={
                "error": "Workload is temporarily at capacity",
                "error_code": "WORKLOAD_OVERLOADED",
                "retryable": True,
                "retry_after_seconds": RETRY_AFTER_SEC,
                "request_id": request_id,
            },
        )

    if isinstance(exc, TranscriptionError):
        return JSONResponse(
            status_code=422,
            content={
                "error": str(exc),
                "error_code": "TRANSCRIPTION_FAILED",
                "retryable": False,
                "request_id": request_id,
            },
        )

    if isinstance(exc, ExtractionError):
        return JSONResponse(
            status_code=422,
            content={
                "error": str(exc),
                "error_code": "EXTRACTION_FAILED",
                "retryable": False,
                "request_id": request_id,
            },
        )

    if isinstance(exc, EmbeddingError):
        return JSONResponse(
            status_code=422,
            content={
                "error": str(exc),
                "error_code": "EMBEDDING_FAILED",
                "retryable": False,
                "request_id": request_id,
            },
        )

    if isinstance(exc, LLMError):
        return JSONResponse(
            status_code=502,
            content={
                "error": str(exc),
                "error_code": "LLM_ERROR",
                "retryable": True,
                "retry_after_seconds": 5,
                "request_id": request_id,
            },
        )

    logger.exception("unhandled_error", request_id=request_id, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "retryable": True,
            "retry_after_seconds": 5,
            "request_id": request_id,
        },
    )
