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
    "rerank": 1,
    "extraction": 4,
    "llm_sync": 4,
}
ADMISSION_WAIT_SEC = 0.05
RETRY_AFTER_SEC = 1
T = TypeVar("T")


class WorkloadOverloadedError(Exception):
    def __init__(self, workload: str) -> None:
        self.workload = workload
        super().__init__(f"{workload} workload is at capacity")


class WorkloadAdmission:
    def __init__(self, capacities: dict[str, int] | None = None) -> None:
        configured = capacities or WORKLOAD_CAPACITY
        self._semaphores = {
            workload: asyncio.BoundedSemaphore(capacity)
            for workload, capacity in configured.items()
        }

    @asynccontextmanager
    async def acquire(self, workload: str) -> AsyncIterator[None]:
        semaphore = self._semaphores[workload]
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=ADMISSION_WAIT_SEC)
        except TimeoutError as exc:
            workload_admission_total.labels(workload=workload, outcome="rejected").inc()
            raise WorkloadOverloadedError(workload) from exc

        workload_admission_total.labels(workload=workload, outcome="accepted").inc()
        workload_in_flight.labels(workload=workload).inc()
        try:
            yield
        finally:
            workload_in_flight.labels(workload=workload).dec()
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
