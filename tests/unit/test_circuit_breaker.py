import asyncio

import httpx
import pytest

from src.common.clients.circuit_breaker import CircuitBreaker, CircuitState
from src.common.clients.cms import _is_countable_cms_failure
from src.common.middleware.error_handler import CircuitOpenError


@pytest.fixture
def breaker() -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=3, reset_timeout_sec=1, half_open_requests=2)


async def _success() -> str:
    return "ok"


async def _failure() -> str:
    raise ConnectionError("connection refused")


@pytest.mark.asyncio
async def test_starts_closed(breaker: CircuitBreaker) -> None:
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_success_keeps_closed(breaker: CircuitBreaker) -> None:
    result = await breaker.execute(_success)
    assert result == "ok"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_opens_after_threshold(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        with pytest.raises(ConnectionError):
            await breaker.execute(_failure)

    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_rejects_when_open(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        with pytest.raises(ConnectionError):
            await breaker.execute(_failure)

    with pytest.raises(CircuitOpenError):
        await breaker.execute(_success)


@pytest.mark.asyncio
async def test_half_open_after_timeout(breaker: CircuitBreaker) -> None:
    import time

    for _ in range(3):
        with pytest.raises(ConnectionError):
            await breaker.execute(_failure)

    assert breaker.state == CircuitState.OPEN

    # Manually set last_failure_time to simulate timeout
    breaker._last_failure_time = time.monotonic() - 2

    result = await breaker.execute(_success)
    assert result == "ok"
    assert breaker.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_closes_after_half_open_successes(breaker: CircuitBreaker) -> None:
    import time

    for _ in range(3):
        with pytest.raises(ConnectionError):
            await breaker.execute(_failure)

    breaker._last_failure_time = time.monotonic() - 2

    await breaker.execute(_success)
    assert breaker.state == CircuitState.HALF_OPEN

    await breaker.execute(_success)
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_reopens_on_half_open_failure(breaker: CircuitBreaker) -> None:
    import time

    for _ in range(3):
        with pytest.raises(ConnectionError):
            await breaker.execute(_failure)

    breaker._last_failure_time = time.monotonic() - 2

    await breaker.execute(_success)
    assert breaker.state == CircuitState.HALF_OPEN

    with pytest.raises(ConnectionError):
        await breaker.execute(_failure)
    assert breaker.state == CircuitState.OPEN


def test_cms_4xx_does_not_count_but_overload_and_5xx_do() -> None:
    request = httpx.Request("GET", "http://cms.test/internal/example")
    for status in (400, 401, 403, 404, 409, 422):
        response = httpx.Response(status, request=request)
        error = httpx.HTTPStatusError("bad request", request=request, response=response)
        assert not _is_countable_cms_failure(error)
    for status in (429, 500, 503):
        response = httpx.Response(status, request=request)
        error = httpx.HTTPStatusError("unavailable", request=request, response=response)
        assert _is_countable_cms_failure(error)


@pytest.mark.asyncio
async def test_half_open_admits_only_configured_parallel_probes(breaker: CircuitBreaker) -> None:
    import time

    for _ in range(3):
        with pytest.raises(ConnectionError):
            await breaker.execute(_failure)
    breaker._last_failure_time = time.monotonic() - 2
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_success() -> str:
        started.set()
        await release.wait()
        return "ok"

    first = asyncio.create_task(breaker.execute(blocked_success))
    second = asyncio.create_task(breaker.execute(blocked_success))
    await started.wait()
    with pytest.raises(CircuitOpenError):
        await breaker.execute(_success)
    release.set()
    assert await first == "ok"
    assert await second == "ok"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_cancelled_half_open_probe_releases_its_permit(breaker: CircuitBreaker) -> None:
    import time

    breaker.half_open_requests = 1
    for _ in range(3):
        with pytest.raises(ConnectionError):
            await breaker.execute(_failure)
    breaker._last_failure_time = time.monotonic() - 2
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_probe() -> str:
        started.set()
        await release.wait()
        return "ok"

    probe = asyncio.create_task(breaker.execute(blocked_probe))
    await started.wait()
    probe.cancel()
    with pytest.raises(asyncio.CancelledError):
        await probe

    assert await breaker.execute(_success) == "ok"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_stale_half_open_success_cannot_close_reopened_circuit() -> None:
    import time

    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_sec=1, half_open_requests=2)
    with pytest.raises(ConnectionError):
        await breaker.execute(_failure)
    breaker._last_failure_time = time.monotonic() - 2
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_success() -> str:
        started.set()
        await release.wait()
        return "ok"

    stale_probe = asyncio.create_task(breaker.execute(slow_success))
    await started.wait()
    with pytest.raises(ConnectionError):
        await breaker.execute(_failure)
    assert breaker.state == CircuitState.OPEN

    release.set()
    assert await stale_probe == "ok"
    assert breaker.state == CircuitState.OPEN
