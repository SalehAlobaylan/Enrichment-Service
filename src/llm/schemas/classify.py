from pydantic import BaseModel, Field

MAX_CLASSIFY_ACCOUNTS = 50
MAX_HANDLE_CHARS = 100
MAX_ACCOUNT_NAME_CHARS = 200
MAX_ACCOUNT_BIO_CHARS = 2_000


class AccountToClassify(BaseModel):
    handle: str = Field(..., min_length=1, max_length=MAX_HANDLE_CHARS)
    name: str = Field(default="", max_length=MAX_ACCOUNT_NAME_CHARS)
    bio: str = Field(default="", max_length=MAX_ACCOUNT_BIO_CHARS)


class AccountClassifyRequest(BaseModel):
    # The ambiguous accounts the deterministic pass could not confidently class.
    accounts: list[AccountToClassify] = Field(
        ..., min_length=1, max_length=MAX_CLASSIFY_ACCOUNTS
    )


class AccountClassResult(BaseModel):
    handle: str
    source_class: str  # official | news | person | other


class AccountClassifyResponse(BaseModel):
    results: list[AccountClassResult] = []
