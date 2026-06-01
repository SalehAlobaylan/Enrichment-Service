from pydantic import BaseModel, Field


class TopicLabelRequest(BaseModel):
    # One or more representative article snippets (title + excerpt) that seed a
    # new topic. The label is generated in the same language as the snippets.
    texts: list[str] = Field(..., min_length=1)


class TopicLabelResponse(BaseModel):
    label: str
