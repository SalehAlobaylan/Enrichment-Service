from typing import Literal

from pydantic import BaseModel, Field

# Same values used by TranscribeResponse — see schemas/transcribe.py.
WriteBackStatus = Literal["not_attempted", "ok", "failed"]


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    content_ids: list[str] | None = None


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int
    # When content_ids are supplied, Enrichment also writes each vector to CMS.
    # These fields report that side effect so the caller can detect silent
    # CMS failures (the embedding response body itself stays successful so
    # the vector is still usable for nearest-neighbor lookups).
    write_back_status: WriteBackStatus = "not_attempted"
    write_back_error: str | None = None


class EmbedQueryRequest(BaseModel):
    text: str = Field(..., min_length=1)


class EmbedQueryResponse(BaseModel):
    embedding: list[float]
    model: str
    dimensions: int
