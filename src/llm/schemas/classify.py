from pydantic import BaseModel, Field


class AccountToClassify(BaseModel):
    handle: str
    name: str = ""
    bio: str = ""


class AccountClassifyRequest(BaseModel):
    # The ambiguous accounts the deterministic pass could not confidently class.
    accounts: list[AccountToClassify] = Field(..., min_length=1)


class AccountClassResult(BaseModel):
    handle: str
    source_class: str  # official | news | person | other


class AccountClassifyResponse(BaseModel):
    results: list[AccountClassResult] = []
