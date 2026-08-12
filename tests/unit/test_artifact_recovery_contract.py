import pytest
from pydantic import ValidationError

from src.common.schemas.artifact_recovery import ArtifactRecoveryCorrelation
from src.common.schemas.pipeline_repair import PipelineRepairCorrelation
from src.retrieval.schemas.embed import EmbedRequest


def correlation() -> ArtifactRecoveryCorrelation:
    return ArtifactRecoveryCorrelation(
        request_id="11111111-1111-4111-8111-111111111111",
        attempt_id="22222222-2222-4222-8222-222222222222",
        claim_token="33333333-3333-4333-8333-333333333333",
        fence_token="44444444-4444-4444-8444-444444444444",
        input_digest="a" * 64,
        producer_event_id="enrichment:22222222-2222-4222-8222-222222222222",
    )


def test_recovery_embed_requires_one_exact_content_target() -> None:
    request = EmbedRequest(texts=["body"], content_ids=["item-a"], artifact_recovery=correlation())
    assert request.content_ids == ["item-a"]
    with pytest.raises(ValidationError):
        EmbedRequest(texts=["one", "two"], content_ids=["a", "b"], artifact_recovery=correlation())


def test_recovery_correlation_rejects_missing_fence() -> None:
    payload = correlation().model_dump()
    payload["fence_token"] = ""
    with pytest.raises(ValidationError):
        ArtifactRecoveryCorrelation(**payload)


def test_recovery_correlation_rejects_malformed_identity_and_digest() -> None:
    payload = correlation().model_dump()
    payload["request_id"] = "x" * 36
    with pytest.raises(ValidationError):
        ArtifactRecoveryCorrelation(**payload)
    payload = correlation().model_dump()
    payload["input_digest"] = "G" * 64
    with pytest.raises(ValidationError):
        ArtifactRecoveryCorrelation(**payload)


def test_pipeline_repair_embedding_requires_one_exact_target_and_no_artifact_recovery() -> None:
    pipeline_correlation = PipelineRepairCorrelation(
        repair_id="11111111-1111-4111-8111-111111111111",
        attempt_id="22222222-2222-4222-8222-222222222222",
        claim_token="33333333-3333-4333-8333-333333333333",
        fence_token="44444444-4444-4444-8444-444444444444",
        expected_item_version="2026-08-11T12:00:00.000000000Z",
        input_digest="b" * 64,
    )
    request = EmbedRequest(texts=["body"], content_ids=["item-a"], pipeline_repair=pipeline_correlation)
    assert request.pipeline_repair == pipeline_correlation
    with pytest.raises(ValidationError):
        EmbedRequest(texts=["one", "two"], content_ids=["a", "b"], pipeline_repair=pipeline_correlation)
    with pytest.raises(ValidationError):
        EmbedRequest(texts=["body"], content_ids=["item-a"], artifact_recovery=correlation(), pipeline_repair=pipeline_correlation)
