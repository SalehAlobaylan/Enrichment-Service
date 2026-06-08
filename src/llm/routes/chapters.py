from fastapi import APIRouter, Depends, Request

from src.common.auth.service_auth import verify_service_token
from src.common.middleware.error_handler import LLMError
from src.common.utils.logging import get_logger
from src.llm.schemas.chapters import ChaptersGenerateRequest, ChaptersGenerateResponse
from src.llm.services.chapters import ChaptersGenerationService

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/chapters/generate", response_model=ChaptersGenerateResponse)
async def generate_chapters(
    body: ChaptersGenerateRequest, request: Request
) -> ChaptersGenerateResponse:
    llm_client = request.app.state.llm_client
    service = ChaptersGenerationService(llm_client)

    try:
        return await service.generate(body)
    except LLMError:
        raise
    except Exception as exc:
        logger.error("chapters_generation_failed", error=str(exc))
        raise LLMError(f"Chapters generation failed: {exc}") from exc
