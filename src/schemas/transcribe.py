from pydantic import BaseModel


class TranscribeSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    segments: list[TranscribeSegment]
    duration_sec: float
