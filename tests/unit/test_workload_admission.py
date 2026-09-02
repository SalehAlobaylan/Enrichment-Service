import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from src.common.workload_admission import (
    WorkloadAdmission,
    WorkloadExecutors,
    WorkloadOverloadedError,
)
from src.retrieval.routes.embed import embed
from src.retrieval.schemas.embed import EmbedRequest


def test_reserved_lane_requires_durable_stage_correlation() -> None:
    with pytest.raises(ValueError, match="reserved embedding lanes"):
        EmbedRequest(texts=["bounded"], content_ids=["item-1"], lane="news")


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
async def test_embed_route_preserves_retryable_overload() -> None:
    class RejectingAdmission:
        @asynccontextmanager
        async def acquire(self, workload: str, lane: str | None = None):
            raise WorkloadOverloadedError(workload)
            yield

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                model_manager=SimpleNamespace(
                    embedder=SimpleNamespace(is_loaded=True),
                ),
                cms_client=SimpleNamespace(),
                llm_client=SimpleNamespace(),
                workload_admission=RejectingAdmission(),
                workload_executors=None,
            ),
        ),
    )

    with pytest.raises(WorkloadOverloadedError, match="embedding workload"):
        await embed(EmbedRequest(texts=["capacity test"]), request)


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
async def test_embedding_lanes_preserve_one_slot_for_waiting_other_lane() -> None:
    admission = WorkloadAdmission({"embedding": 2})
    news_release = asyncio.Event()
    news_started = [asyncio.Event(), asyncio.Event()]
    pods_started = asyncio.Event()

    async def hold_news(index: int) -> None:
        async with admission.acquire("embedding", lane="news"):
            news_started[index].set()
            await news_release.wait()

    first = asyncio.create_task(hold_news(0))
    await news_started[0].wait()
    second = asyncio.create_task(hold_news(1))
    await news_started[1].wait()

    async def enter_pods() -> None:
        async with admission.acquire("embedding", lane="pods"):
            pods_started.set()

    pods = asyncio.create_task(enter_pods())
    await asyncio.sleep(0.005)
    assert not pods_started.is_set()

    news_release.set()
    await asyncio.wait_for(pods_started.wait(), timeout=0.1)
    await asyncio.gather(first, second, pods)


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


@pytest.mark.asyncio
async def test_snapshot_is_attributed_to_one_embedding_lane() -> None:
    admission = WorkloadAdmission({"embedding": 2})
    async with admission.acquire("embedding", lane="news"):
        news = admission.snapshot("news")
        pods = admission.snapshot("pods")

    assert news["embedding_news"]["accepted"] == 1
    assert news["embedding_news"]["in_flight"] == 1
    assert "embedding_news" not in pods
