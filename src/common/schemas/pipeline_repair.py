"""Opaque CMS-issued correlation for one Pipeline repair embedding effect."""

from pydantic import BaseModel, Field


class PipelineRepairCorrelation(BaseModel):
    repair_id: str = Field(min_length=36, max_length=36)
    attempt_id: str = Field(min_length=36, max_length=36)
    claim_token: str = Field(min_length=36, max_length=36)
    fence_token: str = Field(min_length=36, max_length=36)
    expected_item_version: str = Field(min_length=20, max_length=64)
    input_digest: str = Field(min_length=64, max_length=128)
