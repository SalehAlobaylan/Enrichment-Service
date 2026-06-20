from pydantic import BaseModel, Field


class TopicDigestRequest(BaseModel):
    # Representative member posts (title + excerpt/body) of ONE news story, from
    # multiple sources. The digest is generated in the posts' language (Arabic).
    texts: list[str] = Field(..., min_length=1)
    max_bullets: int = Field(default=3, ge=1, le=6)


class TopicDigestResponse(BaseModel):
    # One-line neutral lede + short bullets, grounded ONLY in the posts.
    summary: str
    bullets: list[str]
    # One slug from the finite news taxonomy (see topic_digest service). Always a
    # valid slug; "general" when no category fits or the model output is invalid.
    category: str = "general"
