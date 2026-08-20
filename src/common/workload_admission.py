"""Small, lifecycle-owned admission gates for expensive local work."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Callable, TypeVar

from src.common.utils.metrics import workload_admission_total, workload_in_flight

# Code policy defaults, sized for the constrained split deployments. These are
# intentionally not environment tuning knobs; operators should use replicas or
# a future CMS-backed capacity policy rather than proliferating process config.
WORKLOAD_CAPACITY = {
    "embedding": 2,
    "embedding_query": 1,
    "rerank": 1,
    "extraction": 4,
    "llm_sync": 4,
}
ADMISSION_WAIT_SEC = 0.05
RETRY_AFTER_SEC = 1
T = TypeVar("T")


class WorkloadOverloadedError(Exception):
    def __init__(self, workload: str, lane: str | None = None) -> None:
        self.workload = workload
        self.lane = lane
        super().__init__(f"{workload}{':' + lane if lane else ''} workload is at capacity")


class EmbeddingLaneAdmission:
    """Two-slot scheduler with one non-preemptive reserved floor per lane."""

    def __init__(self, capacity: int = 2) -> None:
        self.capacity = max(1, capacity)
        self._condition = asyncio.Condition()
        self._active = {"news": 0, "pods": 0, "legacy": 0}
        self._waiting = {"news": 0, "pods": 0, "legacy": 0}

    def _can_enter(self, lane: str) -> bool:
        total = sum(self._active.values())
        if total >= self.capacity:
            return False
        if lane == "legacy":
            return self._waiting["news"] == 0 and self._waiting["pods"] == 0
        other = "pods" if lane == "news" else "news"
        if self._active[lane] == 0:
            return True
        return self._waiting[other] == 0

    async def enter(self, lane: str) -> None:
        if lane not in self._active:
            raise ValueError("embedding lane must be news or pods")
        async with self._condition:
            self._waiting[lane] += 1
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(lambda: self._can_enter(lane)),
                    timeout=ADMISSION_WAIT_SEC,
                )
                self._active[lane] += 1
            finally:
                self._waiting[lane] -= 1

    async def leave(self, lane: str) -> None:
        async with self._condition:
            self._active[lane] -= 1
            self._condition.notify_all()


class WorkloadAdmission:
    def __init__(self, capacities: dict[str, int] | None = None) -> None:
        configured = capacities or WORKLOAD_CAPACITY
        self._semaphores = {
            workload: asyncio.BoundedSemaphore(capacity)
            for workload, capacity in configured.items()
            if workload != "embedding"
        }
        self._embedding = EmbeddingLaneAdmission(configured.get("embedding", 2))

    @asynccontextmanager
    async def acquire(self, workload: str, lane: str | None = None) -> AsyncIterator[None]:
        metric_workload = f"{workload}_{lane}" if lane else workload
        if workload == "embedding":
            embedding_lane = lane or "legacy"
            try:
                await self._embedding.enter(embedding_lane)
            except TimeoutError as exc:
                workload_admission_total.labels(
                    workload=metric_workload, outcome="rejected"
                ).inc()
                raise WorkloadOverloadedError(workload, lane) from exc
            workload_admission_total.labels(
                workload=metric_workload, outcome="accepted"
            ).inc()
            workload_in_flight.labels(workload=metric_workload).inc()
            try:
                yield
            finally:
                workload_in_flight.labels(workload=metric_workload).dec()
                await self._embedding.leave(embedding_lane)
            return

        semaphore = self._semaphores[workload]
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=ADMISSION_WAIT_SEC)
        except TimeoutError as exc:
            workload_admission_total.labels(workload=metric_workload, outcome="rejected").inc()
            raise WorkloadOverloadedError(workload, lane) from exc

        workload_admission_total.labels(workload=metric_workload, outcome="accepted").inc()
        workload_in_flight.labels(workload=metric_workload).inc()
        try:
            yield
        finally:
            workload_in_flight.labels(workload=metric_workload).dec()
            semaphore.release()


class WorkloadExecutors:
    """Lifecycle-owned thread pools that keep blocking workload classes apart."""

    def __init__(self, capacities: dict[str, int] | None = None) -> None:
        configured = capacities or WORKLOAD_CAPACITY
        self._executors: dict[str, Executor] = {
            workload: ThreadPoolExecutor(
                max_workers=capacity, thread_name_prefix=f"enrichment-{workload}"
            )
            for workload, capacity in configured.items()
        }

    async def run(
        self, workload: str, function: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executors[workload], partial(function, *args, **kwargs)
        )

    def close(self) -> None:
        for executor in self._executors.values():
            executor.shutdown(wait=False, cancel_futures=True)
