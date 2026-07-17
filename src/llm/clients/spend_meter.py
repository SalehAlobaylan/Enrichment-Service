"""Bounded best-effort delivery of idempotent AI-spend events."""
from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

from src.common.utils.logging import get_logger
from src.common.utils.metrics import (
    ai_spend_delivery_retries_total,
    ai_spend_delivery_total,
    ai_spend_queue_depth,
    ai_spend_queue_oldest_age_seconds,
)

logger = get_logger(__name__)

QUEUE_CAPACITY = 1_000
BATCH_SIZE = 25
MAX_DELIVERY_ATTEMPTS = 3
SHUTDOWN_DRAIN_SEC = 3.0


class SpendDeliveryClient(Protocol):
    async def emit_ai_spend_events(self, events: list[dict[str, Any]]) -> None: ...


class SpendMeter:
    """In-memory, at-least-once while process-live; bounded best-effort on crash."""

    def __init__(self, client: SpendDeliveryClient, capacity: int = QUEUE_CAPACITY) -> None:
        self._client = client
        self._queue: asyncio.Queue[tuple[dict[str, Any], float] | None] = asyncio.Queue(
            maxsize=capacity
        )
        self._worker: asyncio.Task[None] | None = None
        self._accepting = True

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="ai-spend-meter")

    def enqueue(self, event: dict[str, Any]) -> None:
        if not self._accepting:
            ai_spend_delivery_total.labels(outcome="dropped_shutdown").inc()
            return
        try:
            self._queue.put_nowait((event, time.monotonic()))
            ai_spend_queue_depth.set(self._queue.qsize())
        except asyncio.QueueFull:
            ai_spend_delivery_total.labels(outcome="dropped_full").inc()
            logger.warning("ai_spend_meter_queue_full")

    async def close(self, drain_sec: float = SHUTDOWN_DRAIN_SEC) -> None:
        self._accepting = False
        if self._worker is None:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=drain_sec)
        except TimeoutError:
            dropped = self._queue.qsize()
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()
            if dropped:
                ai_spend_delivery_total.labels(outcome="dropped_shutdown").inc(dropped)
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
            ai_spend_queue_depth.set(0)
            return
        await self._queue.put(None)
        await self._worker
        self._worker = None
        ai_spend_queue_depth.set(0)

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                self._queue.task_done()
                return
            batch = [first]
            while len(batch) < BATCH_SIZE:
                try:
                    next_event = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if next_event is None:
                    self._queue.task_done()
                    break
                batch.append(next_event)
            try:
                events = [event for event, _ in batch]
                ai_spend_queue_oldest_age_seconds.set(
                    max(0.0, time.monotonic() - min(enqueued_at for _, enqueued_at in batch))
                )
                for attempt in range(MAX_DELIVERY_ATTEMPTS):
                    try:
                        await self._client.emit_ai_spend_events(events)
                        ai_spend_delivery_total.labels(outcome="delivered").inc(len(events))
                        break
                    except Exception as exc:  # noqa: BLE001 - telemetry is best effort
                        if attempt + 1 == MAX_DELIVERY_ATTEMPTS:
                            ai_spend_delivery_total.labels(outcome="dropped_failed").inc(len(events))
                            logger.warning("ai_spend_meter_delivery_failed", error=str(exc))
                        else:
                            ai_spend_delivery_retries_total.inc()
                            await asyncio.sleep(0.1 * (attempt + 1))
            finally:
                for _ in batch:
                    self._queue.task_done()
                ai_spend_queue_depth.set(self._queue.qsize())
