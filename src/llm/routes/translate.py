from fastapi import APIRouter, Depends, Request

from src.common.auth.service_auth import verify_service_token
from src.common.middleware.error_handler import LLMError
from src.common.utils.logging import get_logger
from src.common.utils.metrics import translations_total
from src.llm.schemas.translate import TranslateRequest, TranslateResponse
from src.llm.services.translation import TranslationService

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/translate", response_model=TranslateResponse)
async def translate(body: TranslateRequest, request: Request) -> TranslateResponse:
    llm_client = request.app.state.llm_client
    cms_client = request.app.state.cms_client
    service = TranslationService(llm_client, cms_client)

    try:
        return await service.translate(
            text=body.text,
            target_language=body.target_language,
            source_language=body.source_language,
            content_id=body.content_id,
        )
    except LLMError:
        raise
    except Exception as exc:
        translations_total.labels(status="failure").inc()
        logger.error("translation_failed", error=str(exc))
        raise LLMError(f"Translation failed: {exc}") from exc
