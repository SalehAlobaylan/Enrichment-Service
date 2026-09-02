from pydantic import BaseModel, Field

UUID_PATTERN = (
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)
DIGEST_PATTERN = r"^[0-9a-f]{64}$"
EVENT_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9:._/-]{0,254}$"


class ArtifactRecoveryCorrelation(BaseModel):
    request_id: str = Field(pattern=UUID_PATTERN)
    attempt_id: str = Field(pattern=UUID_PATTERN)
    claim_token: str = Field(pattern=UUID_PATTERN)
    fence_token: str = Field(pattern=UUID_PATTERN)
    input_digest: str = Field(pattern=DIGEST_PATTERN)
    producer_event_id: str = Field(pattern=EVENT_PATTERN)
