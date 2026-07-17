from pydantic import BaseModel, Field, field_validator

MAX_TOPIC_LABEL_TEXTS = 10
MAX_TOPIC_LABEL_TEXT_CHARS = 6_000


class TopicLabelRequest(BaseModel):
    # One or more representative article snippets (title + excerpt) that seed a
    # new topic. The label is generated in the same language as the snippets.
    texts: list[str] = Field(..., min_length=1, max_length=MAX_TOPIC_LABEL_TEXTS)

    @field_validator("texts")
    @classmethod
    def _validate_texts(cls, texts: list[str]) -> list[str]:
        if any(not text.strip() or len(text) > MAX_TOPIC_LABEL_TEXT_CHARS for text in texts):
            raise ValueError("texts must be non-empty and at most 6000 characters")
        return texts


class TopicLabelResponse(BaseModel):
    label: str
