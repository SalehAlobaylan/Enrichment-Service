import asyncio

import pytest

from src.llm.clients.spend_meter import SpendMeter


class FakeDelivery:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.batches: list[list[dict]] = []

    async def emit_ai_spend_events(self, events: list[dict]) -> None:
        self.batches.append(events)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("temporary CMS outage")


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
        async def emit_ai_spend_events(self, events: list[dict]) -> None:
            await asyncio.Event().wait()

    meter = SpendMeter(SlowDelivery())
    meter.start()
    meter.enqueue({"event_id": "one"})
    await meter.close(drain_sec=0.001)
