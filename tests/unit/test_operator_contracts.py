from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from src.operator.schemas import (
    CONTRACT_VERSION,
    OperatorDecisionPacket,
    OperatorReasonRequest,
    OperatorReasonResponse,
)


def packet_data() -> dict:
    now = datetime.now(UTC)
    return {
        "schema_version": CONTRACT_VERSION,
        "packet_id": "packet-1",
        "fingerprint": "hash",
        "tenant_id": "tenant-a",
        "actor_id": "admin-a",
        "visible_context": {"domain": "media_sources", "view": "list"},
        "collection_started_at": now,
        "collection_ended_at": now,
        "completeness": "complete",
        "evidence": [
            {
                "evidence_id": "ev-1",
                "authority": "live",
                "domain": "media_sources",
                "adapter_key": "media_sources",
                "adapter_version": "v1",
                "tenant_id": "tenant-a",
                "required_permission": "source:read",
                "record_refs": [],
                "deep_link": "/platform/media/sources",
                "observed_at": now,
                "fetched_at": now,
                "max_age_seconds": 60,
                "expires_at": now + timedelta(minutes=1),
                "content_hash": "hash",
                "source_version": "1",
                "availability": "available",
            }
        ],
        "facts": [{"key": "pending_count", "value": 1, "evidence_ids": ["ev-1"]}],
    }


def reason_request() -> OperatorReasonRequest:
    return OperatorReasonRequest.model_validate(
        {
            "schema_version": CONTRACT_VERSION,
            "investigation_id": "investigation-1",
            "thread_id": "thread-1",
            "task_kind": "explain",
            "language": "en",
            "tier": "fast",
            "admin_message": "Why is this pending?",
            "packet": packet_data(),
            "advertised_recommendation_ids": ["recommendation-1"],
            "advertised_actions": [{"key": "source.retry", "target_ids": ["source-1"]}],
            "data_categories": ["operational_state"],
        }
    )


def test_operator_packet_rejects_credentials_and_cross_tenant_evidence() -> None:
    data = packet_data()
    data["visible_context"]["token"] = "never-send"
    with pytest.raises(ValidationError):
        OperatorDecisionPacket.model_validate(data)

    data = packet_data()
    data["evidence"][0]["tenant_id"] = "tenant-b"
    with pytest.raises(ValidationError):
        OperatorDecisionPacket.model_validate(data)


def test_operator_response_cannot_invent_action_or_evidence() -> None:
    request = reason_request()
    response = OperatorReasonResponse.model_validate(
        {
            "schema_version": CONTRACT_VERSION,
            "language": "en",
            "task_kind": "explain",
            "blocks": [
                {"kind": "fact", "text": "One item is pending.", "evidence_ids": ["missing"]}
            ],
            "provider": "test",
            "model": "test",
            "tier": "fast",
            "fallback_used": False,
            "cache": "disabled",
            "action_intent": {"action_key": "source.delete", "target_ids": ["source-1"]},
        }
    )
    with pytest.raises(ValueError):
        response.validate_against_request(request)
