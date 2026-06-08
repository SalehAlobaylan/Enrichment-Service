from typing import Literal

from pydantic import BaseModel, Field


class ChapterWindow(BaseModel):
    """A coarse time-window of the transcript (CMS groups fine segments into
    ~N-second windows before sending). The LLM picks BOUNDARIES by window
    index, never by raw timestamp — so chapters snap to real times."""

    index: int
    start_sec: float
    text: str


class ChaptersGenerateRequest(BaseModel):
    windows: list[ChapterWindow] = Field(..., min_length=1)
    # auto = natural/variable count; count = ~target_count; duration = ~target each.
    mode: Literal["auto", "count", "duration"] = "auto"
    target_count: int | None = None
    target_duration_sec: int | None = None
    min_sec: int | None = None
    max_sec: int | None = None
    with_summary: bool = True
    language: str | None = None


class GeneratedChapter(BaseModel):
    # Index into the request's `windows` where this chapter STARTS. CMS maps it
    # back to windows[start_index].start_sec.
    start_index: int
    title: str
    summary: str | None = None


class ChaptersGenerateResponse(BaseModel):
    chapters: list[GeneratedChapter]
