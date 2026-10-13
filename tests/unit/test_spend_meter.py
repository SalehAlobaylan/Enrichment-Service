import asyncio

import pytest

from src.llm.clients.spend_meter import SpendMeter


class FakeDelivery:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.batches: list[list[dict]] = []
        self.health: list[dict] = []

    async def emit_ai_spend_events(self, events: list[dict], health=None) -> dict:
        if health is not None:
            self.health.append(health)
        if not events:
            return {}
        self.batches.append(events)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("temporary CMS outage")
        return {
            "acknowledgements": [{"event_id": e["event_id"], "status": "inserted"} for e in events]
        }


@pytest.mark.asyncio
async def test_spend_meter_batches_and_retries_same_event_ids() -> None:
    delivery = FakeDelivery(failures=1)
    meter = SpendMeter(delivery)
    meter.enqueue({"event_id": "one"})
    meter.enqueue({"event_id": "two"})
    meter.start()
    await meter.close()

    assert [event["event_id"] for event in delivery.batches[0]] == ["one", "two"]
    assert [event["event_id"] for event in delivery.batches[1]] == ["one", "two"]


@pytest.mark.asyncio
async def test_spend_meter_drops_when_full_without_blocking_caller() -> None:
    delivery = FakeDelivery()
    meter = SpendMeter(delivery, capacity=1)
    meter.enqueue({"event_id": "one"})
    meter.enqueue({"event_id": "two"})
    assert meter._queue.qsize() == 1
    meter.start()
    await meter.close()


@pytest.mark.asyncio
async def test_spend_meter_shutdown_deadline_is_bounded() -> None:
    class SlowDelivery(FakeDelivery):
        async def emit_ai_spend_events(self, events: list[dict], health=None) -> dict:
            await asyncio.Event().wait()

    meter = SpendMeter(SlowDelivery())
    meter.start()
    meter.enqueue({"event_id": "one"})
    await meter.close(drain_sec=0.001)


@pytest.mark.asyncio
async def test_partial_acknowledgements_retry_only_unresolved_ids() -> None:
    class Partial(FakeDelivery):
        async def emit_ai_spend_events(self, events, health=None):
            if not events:
                return {}
            self.batches.append(events)
            if len(self.batches) == 1:
                return {
                    "acknowledgements": [
                        {"event_id": "one", "status": "inserted"},
                        {"event_id": "bad", "status": "invalid"},
                    ]
                }
            return {"acknowledgements": [{"event_id": "two", "status": "duplicate"}]}

    delivery = Partial()
    meter = SpendMeter(delivery)
    for id in ["one", "two", "bad"]:
        meter.enqueue({"event_id": id, "tenant_id": "a"})
    meter.start()
    await meter.close()
    assert [e["event_id"] for e in delivery.batches[1]] == ["two"]
    assert meter._counts["a"]["accepted"] == 2 and meter._counts["a"]["dropped"] == 1


@pytest.mark.asyncio
async def test_http_success_without_durable_ack_is_loss_and_occurrence_is_stamped() -> None:
    class Missing(FakeDelivery):
        async def emit_ai_spend_events(self, events, health=None):
            return {}

    meter = SpendMeter(Missing())
    meter.enqueue({"event_id": "one", "tenant_id": "a"})
    event, _ = meter._queue._queue[0]
    assert event["occurred_at"] and event["event_count"] == 1
    meter.unobservable_attempt("a")
    meter.start()
    await meter.close()
    assert meter._counts["a"]["accepted"] == 0
    assert meter._counts["a"]["dropped"] == 1 and meter._counts["a"]["unobservable"] == 1


@pytest.mark.asyncio
async def test_terminal_health_distinguishes_clean_close_from_abrupt_restart() -> None:
    delivery = FakeDelivery()
    meter = SpendMeter(delivery)
    meter.enqueue({"event_id": "one", "tenant_id": "a"})
    await meter.publish_health()
    assert delivery.health[-1]["closed"] is False
    meter.start()
    await meter.close()
    assert delivery.health[-1]["closed"] is True
    assert delivery.health[-1]["accepted"] == 1
    assert delivery.health[-1]["dropped"] == 0


@pytest.mark.asyncio
async def test_shutdown_does_not_report_closed_before_all_buffered_work_is_accounted() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class PausedDelivery(FakeDelivery):
        async def emit_ai_spend_events(self, events, health=None):
            if events:
                started.set()
                await release.wait()
            return await super().emit_ai_spend_events(events, health)

    delivery = PausedDelivery()
    meter = SpendMeter(delivery)
    meter.enqueue({"event_id": "one", "tenant_id": "a"})
    meter.start()
    await started.wait()
    closing = asyncio.create_task(meter.close())
    await asyncio.sleep(0)
    await meter.publish_health()
    assert delivery.health[-1]["closed"] is False
    release.set()
    await closing
    assert delivery.health[-1]["closed"] is True
    assert delivery.health[-1]["accepted"] == 1
