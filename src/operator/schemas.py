"""Strict, authority-free contracts for Wahb Operator reasoning.

The CMS owns evidence, eligibility, plans, and execution. These models make
that boundary explicit: the reasoning service receives a completed packet and
may only refer to identifiers that CMS already advertised.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACT_VERSION = "wahb-operator/v1"
_FORBIDDEN_KEYS = {
    "password",
    "secret",
    "token",
    "api_key",
    "authorization",
    "cookie",
    "database_url",
    "signing_key",
}


def _has_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).strip().lower() in _FORBIDDEN_KEYS or _has_forbidden_key(child):
                return True
    if isinstance(value, list):
        return any(_has_forbidden_key(child) for child in value)
    return False


class OperatorSubjectRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1, max_length=80)
    id: str = Field(min_length=1, max_length=200)
    label: str | None = Field(default=None, max_length=300)


class OperatorEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1, max_length=200)
    authority: Literal["live", "derived", "temporal", "retrieved", "memory"]
    domain: str = Field(min_length=1, max_length=100)
    adapter_key: str = Field(min_length=1, max_length=150)
    adapter_version: str = Field(min_length=1, max_length=50)
    tenant_id: str = Field(min_length=1, max_length=120)
    required_permission: str = Field(min_length=1, max_length=120)
    record_refs: list[OperatorSubjectRef] = Field(default_factory=list, max_length=200)
    deep_link: str = Field(pattern=r"^/")
    observed_at: datetime
    fetched_at: datetime
    max_age_seconds: int = Field(gt=0, le=3600)
    expires_at: datetime
    content_hash: str = Field(min_length=1, max_length=200)
    source_version: str = Field(min_length=1, max_length=100)
    availability: Literal["available", "partial", "stale", "unavailable", "conflicting"]

    @model_validator(mode="after")
    def validate_freshness(self) -> "OperatorEvidenceRef":
        if self.expires_at < self.fetched_at:
            raise ValueError("expires_at must not precede fetched_at")
        return self


class OperatorFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=150)
    value: Any
    evidence_ids: list[str] = Field(min_length=1, max_length=200)


class OperatorRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=160)
    kind: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=2000)
    deep_link: str = Field(pattern=r"^/")
    evidence_ids: list[str] = Field(min_length=1, max_length=20)
    manual_only: bool

    @field_validator("deep_link")
    @classmethod
    def deep_link_is_same_origin(cls, value: str) -> str:
        if value.startswith("//"):
            raise ValueError("recommendation deep link must be same-origin")
        return value


class OperatorDecisionPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[CONTRACT_VERSION]
    packet_id: str = Field(min_length=1, max_length=200)
    fingerprint: str = Field(min_length=1, max_length=200)
    tenant_id: str = Field(min_length=1, max_length=120)
    actor_id: str = Field(min_length=1, max_length=200)
    visible_context: dict[str, Any]
    collection_started_at: datetime
    collection_ended_at: datetime
    completeness: Literal["complete", "partial"]
    facts: list[OperatorFact] = Field(default_factory=list, max_length=500)
    evidence: list[OperatorEvidenceRef] = Field(default_factory=list, max_length=500)
    recommendations: list[OperatorRecommendation] = Field(default_factory=list, max_length=20)
    warnings: list[str] = Field(default_factory=list, max_length=100)
    unknowns: list[str] = Field(default_factory=list, max_length=100)
    conflicts: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("visible_context")
    @classmethod
    def visible_context_is_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        if _has_forbidden_key(value):
            raise ValueError("visible context contains prohibited credential material")
        return value

    @model_validator(mode="after")
    def validate_packet_references(self) -> "OperatorDecisionPacket":
        if self.collection_ended_at < self.collection_started_at:
            raise ValueError("collection end must not precede collection start")
        evidence_ids = {item.evidence_id for item in self.evidence}
        for item in self.evidence:
            if item.tenant_id != self.tenant_id:
                raise ValueError("evidence tenant must match packet tenant")
        for fact in self.facts:
            if not set(fact.evidence_ids).issubset(evidence_ids):
                raise ValueError("facts may only reference evidence in this packet")
            if _has_forbidden_key(fact.value):
                raise ValueError("facts contain prohibited credential material")
        if len({item.id for item in self.recommendations}) != len(self.recommendations):
            raise ValueError("recommendation IDs must be unique")
        for recommendation in self.recommendations:
            if not set(recommendation.evidence_ids).issubset(evidence_ids):
                raise ValueError("recommendations may only reference evidence in this packet")
        return self


class OperatorRecommendedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=150)
    target_ids: list[str] = Field(min_length=1, max_length=20)


class OperatorReasonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[CONTRACT_VERSION]
    investigation_id: str = Field(min_length=1, max_length=200)
    thread_id: str = Field(min_length=1, max_length=200)
    task_kind: Literal["explain", "investigate", "recommend", "resolve", "compare"]
    language: Literal["ar", "en"]
    tier: Literal["fast", "reasoning"]
    admin_message: str = Field(min_length=1, max_length=8000)
    conversation_objective: str = Field(default="", max_length=4000)
    packet: OperatorDecisionPacket
    advertised_recommendation_ids: list[str] = Field(default_factory=list, max_length=20)
    advertised_actions: list[OperatorRecommendedAction] = Field(default_factory=list, max_length=20)
    data_categories: list[str] = Field(default_factory=list, max_length=30)
    credential_redaction_count: int = Field(default=0, ge=0)


class OperatorResponseBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["fact", "interpretation", "unknown", "recommendation", "degraded"]
    text: str = Field(min_length=1, max_length=4000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=200)


class OperatorActionIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_key: str = Field(min_length=1, max_length=150)
    target_ids: list[str] = Field(min_length=1, max_length=20)


class OperatorReasonModelOutput(BaseModel):
    """Strict model output before service-owned provenance is attached."""

    model_config = ConfigDict(extra="forbid")

    language: Literal["ar", "en"]
    task_kind: Literal["explain", "investigate", "recommend", "resolve", "compare"]
    blocks: list[OperatorResponseBlock] = Field(min_length=1, max_length=30)
    primary_recommendation_id: str | None = None
    secondary_recommendation_ids: list[str] = Field(default_factory=list, max_length=3)
    action_intent: OperatorActionIntent | None = None
    uncertainties: list[str] = Field(default_factory=list, max_length=20)


class OperatorReasonResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[CONTRACT_VERSION]
    language: Literal["ar", "en"]
    task_kind: Literal["explain", "investigate", "recommend", "resolve", "compare"]
    blocks: list[OperatorResponseBlock] = Field(min_length=1, max_length=30)
    primary_recommendation_id: str | None = None
    secondary_recommendation_ids: list[str] = Field(default_factory=list, max_length=3)
    action_intent: OperatorActionIntent | None = None
    uncertainties: list[str] = Field(default_factory=list, max_length=20)
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=200)
    tier: Literal["fast", "reasoning"]
    fallback_used: bool
    cache: Literal["disabled"]

    def validate_against_request(self, request: OperatorReasonRequest) -> None:
        evidence_ids = {item.evidence_id for item in request.packet.evidence}
        for block in self.blocks:
            if block.kind in {"fact", "interpretation"} and not block.evidence_ids:
                raise ValueError("facts and interpretations require evidence")
            if not set(block.evidence_ids).issubset(evidence_ids):
                raise ValueError("response references unadvertised evidence")
        if (
            self.primary_recommendation_id
            and self.primary_recommendation_id not in request.advertised_recommendation_ids
        ):
            raise ValueError("response references unadvertised primary recommendation")
        if len(self.secondary_recommendation_ids) > 3 or not set(
            self.secondary_recommendation_ids
        ).issubset(request.advertised_recommendation_ids):
            raise ValueError("response references unadvertised secondary recommendation")
        if self.action_intent:
            advertised = {item.key: set(item.target_ids) for item in request.advertised_actions}
            if self.action_intent.action_key not in advertised or not set(
                self.action_intent.target_ids
            ).issubset(advertised[self.action_intent.action_key]):
                raise ValueError("response references unadvertised action or target")
