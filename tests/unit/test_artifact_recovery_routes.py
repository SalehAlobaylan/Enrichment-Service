import pytest
from pydantic import ValidationError

from src.common.routes.artifact_recovery import (
    LLMMetadataRecoveryRequest,
    TextEmbeddingRecoveryRequest,
)


def correlation() -> dict[str, str]:
    return {
        "request_id": "11111111-1111-4111-8111-111111111111",
        "attempt_id": "22222222-2222-4222-8222-222222222222",
        "claim_token": "33333333-3333-4333-8333-333333333333",
        "fence_token": "44444444-4444-4444-8444-444444444444",
        "input_digest": "a" * 64,
        "producer_event_id": "enrichment:attempt",
    }


def test_owner_routes_accept_one_exact_typed_target() -> None:
    request = TextEmbeddingRecoveryRequest(text="hello", content_id="55555555-5555-4555-8555-555555555555", correlation=correlation())
    assert request.content_id.endswith("5555")
    metadata = LLMMetadataRecoveryRequest(text="hello", content_id=request.content_id, correlation=correlation())
    assert metadata.correlation.input_digest == "a" * 64


def test_owner_routes_reject_extra_provider_or_queue_arguments() -> None:
    with pytest.raises(ValidationError):
        TextEmbeddingRecoveryRequest.model_validate({"text": "hello", "content_id": "55555555-5555-4555-8555-555555555555", "correlation": correlation(), "queue_name": "arbitrary"})
