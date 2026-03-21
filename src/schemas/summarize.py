from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_length: int = 200
    style: str = "brief"
    content_id: str | None = None


class SummarizeResponse(BaseModel):
    summary: str
    key_points: list[str]
