from pydantic import BaseModel, Field, field_validator

MAX_TOPIC_TEXTS = 10
MAX_TOPIC_TEXT_CHARS = 6_000


class TopicDigestRequest(BaseModel):
    # Representative member posts (title + excerpt/body) of ONE news story, from
    # multiple sources. The digest is generated in the posts' language (Arabic).
    texts: list[str] = Field(..., min_length=1, max_length=MAX_TOPIC_TEXTS)
    max_bullets: int = Field(default=3, ge=1, le=6)

    @field_validator("texts")
    @classmethod
    def _validate_texts(cls, texts: list[str]) -> list[str]:
        if any(not text.strip() or len(text) > MAX_TOPIC_TEXT_CHARS for text in texts):
            raise ValueError("texts must be non-empty and at most 6000 characters")
        return texts


class TopicDigestResponse(BaseModel):
    # One-line neutral lede + short bullets, grounded ONLY in the posts.
    summary: str
    bullets: list[str]
    # One slug from the finite news taxonomy (see topic_digest service). Always a
    # valid slug; "general" when no category fits or the model output is invalid.
    category: str = "general"
