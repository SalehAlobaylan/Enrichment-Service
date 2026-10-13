"""Zero-priced local compute demand, emitted only by the executing role."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from src.llm.clients.spend_context import SpendContext
from src.llm.clients.spend_meter import SpendMeter


def emit_compute(
    meter: SpendMeter | None,
    context: SpendContext,
    spend_class: str,
    model: str,
    units: dict[str, float | int],
) -> None:
    if meter is None:
        return
    meter.enqueue(
        {
            "event_id": str(uuid4()),
            "occurred_at": datetime.now(UTC).isoformat(),
            "tenant_id": context.tenant_id,
            "human_actor_id": context.human_actor_id or "",
            "trigger_source": context.trigger_source,
            "system_run_id": context.system_run_id or "",
            "spend_class": spend_class,
            "operation": spend_class,
            "provider": "local",
            "model": model,
            "units": units,
            "source_service": "enrichment",
        }
    )
