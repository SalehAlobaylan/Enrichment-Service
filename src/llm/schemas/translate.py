from pydantic import BaseModel, Field

MAX_TRANSLATION_TEXT_CHARS = 12_000


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TRANSLATION_TEXT_CHARS)
    source_language: str | None = Field(default=None, min_length=2, max_length=16)
    target_language: str = Field(default="en", min_length=2, max_length=16)
    content_id: str | None = Field(default=None, min_length=1, max_length=128)


class TranslateResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str
    write_back_status: str = "not_attempted"
    write_back_error: str | None = None
