from fastapi import APIRouter, Depends, Request

from src.common.auth.service_auth import verify_service_token
from src.operator.schemas import OperatorReasonRequest, OperatorReasonResponse
from src.operator.services.reasoning import OperatorReasoningService

router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/reason", response_model=OperatorReasonResponse)
async def reason(body: OperatorReasonRequest, request: Request) -> OperatorReasonResponse:
    service = OperatorReasoningService(request.app.state.llm_client)
    return await service.reason(body)
