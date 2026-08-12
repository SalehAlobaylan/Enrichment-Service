from fastapi import APIRouter, Depends, Request

from src.common.auth.service_auth import verify_service_token
from src.common.middleware.error_handler import LLMError
from src.common.utils.logging import get_logger
from src.common.utils.metrics import summarizations_total
from src.llm.schemas.summarize import SummarizeRequest, SummarizeResponse
from src.llm.services.summarization import SummarizationService

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(body: SummarizeRequest, request: Request) -> SummarizeResponse:
    llm_client = request.app.state.llm_client
    cms_client = request.app.state.cms_client
    service = SummarizationService(llm_client, cms_client)

    try:
        return await service.summarize(
            text=body.text,
            max_length=body.max_length,
            style=body.style,
            content_id=body.content_id,
            artifact_recovery=body.artifact_recovery.model_dump()
            if body.artifact_recovery
            else None,
        )
    except LLMError:
        raise
    except Exception as exc:
        summarizations_total.labels(status="failure").inc()
        logger.error("summarization_failed", error=str(exc))
        raise LLMError(f"Summarization failed: {exc}") from exc
