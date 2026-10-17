"""Immutable provenance passed by the owning CMS or Aggregation workflow."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.common.middleware.error_handler import BudgetCappedError


class SpendContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    tenant_id: str = Field(default="default", min_length=1, max_length=64)
    trigger_source: str = Field(default="unknown", min_length=1, max_length=64)
    system_run_id: str | None = Field(default=None, max_length=96)
    override_id: str | None = Field(default=None, max_length=36)
    human_actor_id: str | None = Field(default=None, max_length=160)
    economics_attribution_ref: str | None = Field(default=None, max_length=160)
    serving_exempt: bool = False
    admission: dict[str, Any] | None = None


class SpendRequest(BaseModel):
    spend_context: SpendContext = Field(default_factory=SpendContext)


class SpendAllowanceCache:
    """Refresh off the request path, keyed by tenant and class. Failure is visible."""

    def __init__(self, client: Any, meter: Any = None) -> None:
        self.client = client
        self.meter = meter
        self._entries: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
        self._tasks: dict[tuple[str, str], asyncio.Task[None]] = {}

    async def _refresh(self, key: tuple[str, str]) -> None:
        try:
            answer = await self.client.get_ai_spend_allowance(key[0], key[1])
        except Exception:
            answer = dict(allowed=True, reason="governor_unavailable", ledger_health="unavailable")
        self._entries[key] = (time.monotonic(), answer)

    def check(self, context: SpendContext, spend_class: str = "llm") -> dict[str, Any]:
        key = (context.tenant_id, spend_class)
        entry = self._entries.get(key)
        if entry is None or time.monotonic() - entry[0] >= 60:
            task = self._tasks.get(key)
            if task is None or task.done():
                self._tasks[key] = asyncio.create_task(self._refresh(key))
        if entry is None or time.monotonic() - entry[0] >= 60:
            return dict(allowed=True, reason="governor_unavailable", ledger_health="unavailable")
        return entry[1]

    async def authorize(self, context: SpendContext) -> None:
        # Exemptions are a fixed registry, never a caller-provided bypass bit.
        if context.serving_exempt and context.trigger_source in {"serving_news_summary"}:
            return
        answer = context.admission or self.check(context)
        if answer.get("reason") == "governor_unavailable" and self.meter is not None:
            self.meter.guard_unavailable(context.tenant_id)
        if not answer.get("allowed", True):
            if context.override_id:
                try:
                    answer = await self.client.get_ai_spend_allowance(
                        context.tenant_id, "llm", context.trigger_source, context.override_id
                    )
                except Exception:
                    if self.meter is not None:
                        self.meter.guard_unavailable(context.tenant_id)
                    answer = dict(
                        allowed=True, reason="governor_unavailable", ledger_health="unavailable"
                    )
            if not answer.get("allowed", True):
                raise BudgetCappedError(str(answer.get("reason", "budget_capped")))

    async def close(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
