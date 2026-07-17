"""Reject declared oversized request bodies before FastAPI parses JSON."""
from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

MAX_REQUEST_BODY_BYTES = 1_000_000


class RequestBodyLimitMiddleware:
    """Enforce declared and chunked body limits before route/model work."""

    def __init__(self, app: ASGIApp, max_body_bytes: int = MAX_REQUEST_BODY_BYTES) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode().lower(): value.decode() for key, value in scope["headers"]}
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = self.max_body_bytes + 1
            if declared_size > self.max_body_bytes:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "error": "Request body is too large",
                        "error_code": "REQUEST_BODY_TOO_LARGE",
                        "retryable": False,
                    },
                )
                await response(scope, receive, send)
                return

        received_bytes = 0
        response_started = False

        async def send_tracking(message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise _RequestBodyTooLargeError
            return message

        try:
            await self.app(scope, limited_receive, send_tracking)
        except _RequestBodyTooLargeError:
            if response_started:
                raise
            response = JSONResponse(
                status_code=413,
                content={
                    "error": "Request body is too large",
                    "error_code": "REQUEST_BODY_TOO_LARGE",
                    "retryable": False,
                },
            )
            await response(scope, receive, send)


class _RequestBodyTooLargeError(Exception):
    pass
