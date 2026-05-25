from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_language: str | None = None
    target_language: str = "en"
    content_id: str | None = None


class TranslateResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str
