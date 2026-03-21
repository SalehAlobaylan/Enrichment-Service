import pytest

from src.clients.circuit_breaker import CircuitBreaker, CircuitState
from src.middleware.error_handler import CircuitOpenError


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
