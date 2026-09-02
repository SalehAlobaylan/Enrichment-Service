from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from src.common.auth.service_auth import verify_service_token
from src.common.middleware.error_handler import EmbeddingError, LLMError
from src.common.schemas.artifact_recovery import UUID_PATTERN, ArtifactRecoveryCorrelation
from src.llm.services.summarization import SummarizationService
from src.retrieval.services.embedding import EmbeddingService

router = APIRouter(dependencies=[Depends(verify_service_token)])


class TextEmbeddingRecoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=12_000)
    content_id: str = Field(pattern=UUID_PATTERN)
    correlation: ArtifactRecoveryCorrelation


class LLMMetadataRecoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=12_000)
    content_id: str = Field(pattern=UUID_PATTERN)
    correlation: ArtifactRecoveryCorrelation


class ArtifactRecoveryResult(BaseModel):
    artifact: Literal["text_embedding", "llm_metadata"]
    write_back_status: Literal["ok", "failed", "not_attempted"]


@router.post("/text-embedding", response_model=ArtifactRecoveryResult)
async def recover_text_embedding(
    body: TextEmbeddingRecoveryRequest, request: Request
) -> ArtifactRecoveryResult:
    manager = request.app.state.model_manager
    if not manager.embedder.is_loaded:
        raise EmbeddingError("Embedding model is not loaded")
    service = EmbeddingService(
        manager.embedder,
        request.app.state.cms_client,
        executors=getattr(request.app.state, "workload_executors", None),
    )
    async with request.app.state.workload_admission.acquire("embedding"):
        result = await service.embed(
            [body.text],
            content_ids=[body.content_id],
            artifact_recovery=body.correlation.model_dump(),
        )
    return ArtifactRecoveryResult(
        artifact="text_embedding", write_back_status=result.write_back_status
    )


@router.post("/llm-metadata", response_model=ArtifactRecoveryResult)
async def recover_llm_metadata(
    body: LLMMetadataRecoveryRequest, request: Request
) -> ArtifactRecoveryResult:
    service = SummarizationService(request.app.state.llm_client, request.app.state.cms_client)
    try:
        result = await service.summarize(
            text=body.text,
            max_length=200,
            style="brief",
            content_id=body.content_id,
            artifact_recovery=body.correlation.model_dump(),
        )
    except LLMError:
        raise
    return ArtifactRecoveryResult(
        artifact="llm_metadata", write_back_status=result.write_back_status
    )
