from fastapi import APIRouter, Depends, Request

from src.common.auth.service_auth import verify_service_token
from src.common.middleware.error_handler import LLMError
from src.common.utils.logging import get_logger
from src.llm.schemas.classify import AccountClassifyRequest, AccountClassifyResponse
from src.llm.services.classification import AccountClassificationService

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/classify/accounts", response_model=AccountClassifyResponse)
async def classify_accounts(
    body: AccountClassifyRequest, request: Request
) -> AccountClassifyResponse:
    """Classify ambiguous source accounts (OFFICIAL/NEWS/PERSON/OTHER) the
    deterministic CMS pass could not resolve. Per-account cached LLM call."""
    service = AccountClassificationService(request.app.state.llm_client)
    try:
        results = await service.classify(body.accounts)
        return AccountClassifyResponse(results=results)
    except LLMError:
        raise
    except Exception as exc:
        logger.error("classify_accounts_failed", error=str(exc))
        raise LLMError(f"Account classification failed: {exc}") from exc
