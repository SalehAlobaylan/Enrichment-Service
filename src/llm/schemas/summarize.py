from typing import Literal

from pydantic import BaseModel, Field

from src.common.schemas.artifact_recovery import ArtifactRecoveryCorrelation

MAX_LLM_TEXT_CHARS = 12_000


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_LLM_TEXT_CHARS)
    max_length: int = Field(default=200, ge=20, le=1_000)
    style: Literal["brief", "detailed", "bullet"] = "brief"
    content_id: str | None = Field(default=None, min_length=1, max_length=128)
    artifact_recovery: ArtifactRecoveryCorrelation | None = None


class SummarizeResponse(BaseModel):
    summary: str
    key_points: list[str]
    write_back_status: str = "not_attempted"
    write_back_error: str | None = None
