from fastapi import APIRouter, Depends, Request

from src.common.auth.service_auth import verify_service_token
from src.common.middleware.error_handler import LLMError
from src.common.utils.logging import get_logger
from src.llm.schemas.topic_label import TopicLabelRequest, TopicLabelResponse
from src.llm.services.topic_labeling import TopicLabelingService

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/topics/label", response_model=TopicLabelResponse)
async def topic_label(body: TopicLabelRequest, request: Request) -> TopicLabelResponse:
    llm_client = request.app.state.llm_client
    service = TopicLabelingService(llm_client)

    try:
        return await service.label(texts=body.texts)
    except LLMError:
        raise
    except Exception as exc:
        logger.error("topic_label_failed", error=str(exc))
        raise LLMError(f"Topic labeling failed: {exc}") from exc
