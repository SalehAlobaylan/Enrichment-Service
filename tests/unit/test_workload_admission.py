import asyncio
import threading

import pytest

from src.common.workload_admission import (
    WorkloadAdmission,
    WorkloadExecutors,
    WorkloadOverloadedError,
)


@pytest.mark.asyncio
async def test_saturated_workload_rejects_without_starting_extra_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = WorkloadAdmission({"embedding": 1})
    monkeypatch.setattr("src.common.workload_admission.ADMISSION_WAIT_SEC", 0.001)
    started = asyncio.Event()
    release = asyncio.Event()

    async def hold_capacity() -> None:
        async with admission.acquire("embedding"):
            started.set()
            await release.wait()

    first = asyncio.create_task(hold_capacity())
    await started.wait()
    with pytest.raises(WorkloadOverloadedError, match="embedding workload"):
        async with admission.acquire("embedding"):
            pytest.fail("admission must not start work after rejecting")
    release.set()
    await first


@pytest.mark.asyncio
async def test_cancellation_releases_capacity() -> None:
    admission = WorkloadAdmission({"rerank": 1})
    entered = asyncio.Event()

    async def hold_capacity() -> None:
        async with admission.acquire("rerank"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold_capacity())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with admission.acquire("rerank"):
        pass


@pytest.mark.asyncio
async def test_executors_keep_blocking_workload_threads_isolated() -> None:
    executors = WorkloadExecutors({"embedding": 1, "rerank": 1})
    try:
        embedding_thread = await executors.run("embedding", lambda: threading.current_thread().name)
        rerank_thread = await executors.run("rerank", lambda: threading.current_thread().name)
    finally:
        executors.close()

    assert embedding_thread.startswith("enrichment-embedding")
    assert rerank_thread.startswith("enrichment-rerank")
    assert embedding_thread != rerank_thread
