from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Media Studio Clearance Autopilot (stage 6, Slice 4) — advisory chapter
# proposals. The LLM drafts a publish/reject suggestion for the editorial review
# cases the deterministic rules could not clear. A proposal is NEVER an action:
# CMS re-checks every invariant in code and a human accepts/overrides (S10).


class ChapterProposalItem(BaseModel):
    # Opaque case id echoed back so CMS can pair proposal → action row.
    id: str
    # Chapter start + end transcript slices (already trimmed to the token budget
    # by CMS). Untrusted scraped content — treated as data, never instructions.
    transcript: str = Field(default="", max_length=8000)
    title: str = Field(default="", max_length=400)
    summary: str = Field(default="", max_length=1000)
    review_reason: str = Field(default="", max_length=400)
    review_code: str = Field(default="", max_length=48)
    confidence: float | None = None
    standalone_score: float | None = None
    contains_sponsor: bool = False
    duration_sec: int | None = None
    parent_title: str = Field(default="", max_length=400)


class ChapterProposalRequest(BaseModel):
    items: list[ChapterProposalItem] = Field(..., min_length=1, max_length=15)


class ChapterProposalChecks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_ok: bool = Field(strict=True)
    no_sponsor_overlap: bool = Field(strict=True)
    coherent_start: bool = Field(strict=True)
    coherent_end: bool = Field(strict=True)


class ChapterProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    proposal: Literal["publish", "reject"]
    confidence: float = Field(strict=True, ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=300)
    checked: ChapterProposalChecks


class ChapterProposalResponse(BaseModel):
    # Only well-formed proposals are returned; ids the model failed to produce a
    # valid proposal for are omitted, and CMS ledgers them llm_invalid_output.
    proposals: list[ChapterProposal]
