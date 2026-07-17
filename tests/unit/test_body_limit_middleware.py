import pytest

from src.common.middleware.body_limit import RequestBodyLimitMiddleware


@pytest.mark.asyncio
async def test_chunked_body_exceeding_limit_returns_413_before_app_response() -> None:
    app_started_response = False

    async def app(_scope, receive, send) -> None:
        nonlocal app_started_response
        await receive()
        await receive()
        app_started_response = True
        await send({"type": "http.response.start", "status": 200, "headers": []})

    messages = iter(
        [
            {"type": "http.request", "body": b"a" * 5, "more_body": True},
            {"type": "http.request", "body": b"b" * 6, "more_body": False},
        ]
    )
    sent: list[dict] = []

    async def receive() -> dict:
        return next(messages)

    async def send(message: dict) -> None:
        sent.append(message)

    middleware = RequestBodyLimitMiddleware(app, max_body_bytes=10)
    await middleware({"type": "http", "headers": []}, receive, send)

    assert app_started_response is False
    assert sent[0]["status"] == 413
