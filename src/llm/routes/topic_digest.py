from fastapi import APIRouter, Depends, Request

from src.common.auth.service_auth import verify_service_token
from src.common.middleware.error_handler import LLMError
from src.common.utils.logging import get_logger
from src.llm.schemas.topic_digest import TopicDigestRequest, TopicDigestResponse
from src.llm.services.topic_digest import TopicDigestService

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/topics/digest", response_model=TopicDigestResponse)
async def topic_digest(body: TopicDigestRequest, request: Request) -> TopicDigestResponse:
    llm_client = request.app.state.llm_client
    service = TopicDigestService(llm_client)

    try:
        return await service.digest(texts=body.texts, max_bullets=body.max_bullets)
    except LLMError:
        raise
    except Exception as exc:
        logger.error("topic_digest_failed", error=str(exc))
        raise LLMError(f"Topic digest failed: {exc}") from exc
