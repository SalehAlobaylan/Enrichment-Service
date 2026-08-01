import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.common.middleware.error_handler import LLMError
from src.operator.schemas import CONTRACT_VERSION, OperatorReasonRequest
from src.operator.services.reasoning import OperatorReasoningService


def reason_body() -> dict:
    now = datetime.now(UTC)
    return {
        "schema_version": CONTRACT_VERSION,
        "investigation_id": "investigation-1",
        "thread_id": "thread-1",
        "task_kind": "explain",
        "language": "en",
        "tier": "fast",
        "admin_message": "Why is this pending?",
        "packet": {
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
                    "adapter_key": "media_sources.approval_handoff",
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
        },
    }


@pytest.mark.asyncio
async def test_operator_reasoning_disables_cache_and_attaches_observed_provenance() -> None:
    llm = AsyncMock()
    llm.complete_with_provenance.return_value = SimpleNamespace(
        text=json.dumps(
            {
                "language": "en",
                "task_kind": "explain",
                "blocks": [
                    {"kind": "fact", "text": "One item is pending.", "evidence_ids": ["ev-1"]}
                ],
                "primary_recommendation_id": None,
                "secondary_recommendation_ids": [],
                "action_intent": None,
                "uncertainties": [],
            }
        ),
        provider="test-provider",
        model="test-model",
        fallback_used=True,
        cached=False,
    )
    response = await OperatorReasoningService(llm).reason(
        OperatorReasonRequest.model_validate(reason_body())
    )
    assert response.provider == "test-provider"
    assert response.fallback_used is True
    assert response.cache == "disabled"
    assert llm.complete_with_provenance.await_args.kwargs["allow_cache"] is False
    assert llm.complete_with_provenance.await_args.kwargs["tenant_id"] == "tenant-a"


@pytest.mark.asyncio
async def test_operator_reasoning_rejects_invalid_or_invented_output() -> None:
    llm = AsyncMock()
    llm.complete_with_provenance.return_value = SimpleNamespace(
        text=json.dumps(
            {
                "language": "en",
                "task_kind": "explain",
                "blocks": [
                    {"kind": "fact", "text": "Invented evidence", "evidence_ids": ["not-in-packet"]}
                ],
                "primary_recommendation_id": None,
                "secondary_recommendation_ids": [],
                "action_intent": None,
                "uncertainties": [],
            }
        ),
        provider="test-provider",
        model="test-model",
        fallback_used=False,
        cached=False,
    )
    with pytest.raises(LLMError, match="invalid structured response"):
        await OperatorReasoningService(llm).reason(
            OperatorReasonRequest.model_validate(reason_body())
        )


def test_operator_reason_route_requires_service_token(client) -> None:
    body = OperatorReasonRequest.model_validate(reason_body()).model_dump(mode="json")
    response = client.post("/v1/operator/reason", json=body)
    assert response.status_code == 401
