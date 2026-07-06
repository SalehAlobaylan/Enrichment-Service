from fastapi import APIRouter, Depends, Request

from src.common.auth.service_auth import verify_service_token
from src.common.middleware.error_handler import LLMError
from src.common.utils.logging import get_logger
from src.llm.schemas.chapter_proposal import (
    ChapterProposalRequest,
    ChapterProposalResponse,
)
from src.llm.services.chapter_proposal import ChapterProposalService

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/studio/chapter-proposal", response_model=ChapterProposalResponse)
async def chapter_proposal(
    body: ChapterProposalRequest, request: Request
) -> ChapterProposalResponse:
    llm_client = request.app.state.llm_client
    service = ChapterProposalService(llm_client)

    try:
        return await service.propose_batch(items=body.items)
    except LLMError:
        raise
    except Exception as exc:
        logger.error("chapter_proposal_failed", error=str(exc))
        raise LLMError(f"Chapter proposal failed: {exc}") from exc
