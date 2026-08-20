from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.common.schemas.artifact_recovery import ArtifactRecoveryCorrelation
from src.retrieval.schemas.embed import ContentStageCorrelation

MAX_LLM_TEXT_CHARS = 12_000


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_LLM_TEXT_CHARS)
    max_length: int = Field(default=200, ge=20, le=1_000)
    style: Literal["brief", "detailed", "bullet"] = "brief"
    content_id: str | None = Field(default=None, min_length=1, max_length=128)
    artifact_recovery: ArtifactRecoveryCorrelation | None = None
    content_stage: ContentStageCorrelation | None = None

    @model_validator(mode="after")
    def validate_correlation(self) -> "SummarizeRequest":
        if self.artifact_recovery is not None and self.content_stage is not None:
            raise ValueError("recovery and normal-stage correlations are exclusive")
        if self.content_stage is not None and not self.content_id:
            raise ValueError("content_stage requires content_id")
        return self


class SummarizeResponse(BaseModel):
    summary: str
    key_points: list[str]
    write_back_status: str = "not_attempted"
    write_back_error: str | None = None
