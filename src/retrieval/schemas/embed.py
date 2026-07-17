from typing import Literal

from pydantic import BaseModel, Field, model_validator

MAX_EMBED_BATCH = 32
MAX_EMBED_TEXT_CHARS = 12_000

# Same shape used by other write-back responses across the platform
# (Media-Service's TranscribeResponse / ImageEmbedResponse use the same
# three-value enum). Keep these aligned if the contract changes.
WriteBackStatus = Literal["not_attempted", "ok", "failed"]


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_EMBED_BATCH)
    content_ids: list[str] | None = Field(default=None, max_length=MAX_EMBED_BATCH)
    # When true, Enrichment runs topic-tag + named-entity extraction on the
    # FIRST text (the assumption: callers either send one item per request,
    # or a batch that shares a topic — e.g., multiple chunks of one article).
    # Adds one LLM call per request, so opt-in only.
    extract_tags: bool = False
    # Legacy compatibility flag from the BGE-M3 sparse era. Qwen is dense-only,
    # so this is currently a no-op and sparse write-back remains None.
    extract_sparse: bool = False

    @model_validator(mode="after")
    def _validate_batch(self) -> "EmbedRequest":
        if any(not text.strip() or len(text) > MAX_EMBED_TEXT_CHARS for text in self.texts):
            raise ValueError("texts must be non-empty and at most 12000 characters")
        if self.content_ids is not None:
            if len(self.content_ids) != len(self.texts):
                raise ValueError("content_ids must match texts length")
            if len(set(self.content_ids)) != len(self.content_ids) or any(
                not content_id.strip() for content_id in self.content_ids
            ):
                raise ValueError("content_ids must be non-empty and unique")
        return self


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int
    # Immutable vector-space identity (stage 10). space_id == "" means the space
    # is lifecycle-not-ready (unresolved revision); consumers must not compare a
    # vector whose space_id they cannot confirm.
    space_id: str = ""
    producer_id: str = ""
    # When content_ids are supplied, Enrichment also writes each vector to CMS.
    # These fields report that side effect so the caller can detect silent
    # CMS failures (the embedding response body itself stays successful so
    # the vector is still usable for nearest-neighbor lookups).
    write_back_status: WriteBackStatus = "not_attempted"
    write_back_error: str | None = None
    # Populated when extract_tags=true. None when tagging wasn't requested or
    # the text was too short / the LLM failed (best-effort).
    tags: list[str] | None = None
    entities: dict[str, list[str]] | None = None


class EmbedQueryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_EMBED_TEXT_CHARS)


class EmbedQueryResponse(BaseModel):
    embedding: list[float]
    model: str
    dimensions: int
    # Query-side space identity — the comparability guard requires a query's
    # space_id and only admits candidate rows stamped with the same ID.
    space_id: str = ""
    producer_id: str = ""
