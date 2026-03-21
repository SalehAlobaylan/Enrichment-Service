from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    content_ids: list[str] | None = None


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int


class EmbedQueryRequest(BaseModel):
    text: str = Field(..., min_length=1)


class EmbedQueryResponse(BaseModel):
    embedding: list[float]
    model: str
    dimensions: int
