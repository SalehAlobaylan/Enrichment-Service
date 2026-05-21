import asyncio
import random
import time

from src.config import Settings
from src.middleware.error_handler import LLMError
from src.middleware.request_id import current_request_id
from src.utils.logging import get_logger
from src.utils.metrics import (
    llm_errors_total,
    llm_request_duration,
    llm_requests_total,
    llm_retries_total,
)

logger = get_logger(__name__)

# Retry policy — applied uniformly across providers.
MAX_ATTEMPTS = 3
BASE_BACKOFF_SEC = 1.0  # 1s, 2s, 4s with jitter


def _classify_error(exc: Exception) -> tuple[str, bool]:
    """Return (error_type, is_retryable) for an LLM SDK exception.

    error_type values: rate_limit | timeout | auth | server | bad_request | other
    """
    name = type(exc).__name__
    msg = str(exc).lower()

    # Status-code aware paths (httpx, OpenAI, Anthropic all surface these names)
    status: int | None = getattr(exc, "status_code", None)
    if status is None:
        # Some SDKs nest the status on a .response object
        resp = getattr(exc, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
    if status is None:
        # google-genai's ClientError / ServerError use `.code` for the HTTP status.
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            status = code

    if status is not None:
        if status == 429:
            return "rate_limit", True
        if status in (500, 502, 503, 504):
            return "server", True
        if status in (401, 403):
            return "auth", False
        if status == 400:
            return "bad_request", False

    # Name/string heuristics — covers SDKs that raise typed errors without status
    if "RateLimit" in name or "rate_limit" in msg or "429" in msg:
        return "rate_limit", True
    if "Timeout" in name or "timeout" in msg or "timed out" in msg:
        return "timeout", True
    if "Authentication" in name or "PermissionDenied" in name or "Unauthorized" in name:
        return "auth", False
    if "Connection" in name or "APIConnection" in name:
        return "server", True
    if any(code in msg for code in ("500", "502", "503", "504")):
        return "server", True

    return "other", False


class LLMClient:
    def __init__(self, settings: Settings):
        self.provider = (settings.LLM_PROVIDER or "").strip().lower()
        self.model = settings.LLM_MODEL
        self._openai_client = None
        self._anthropic_client = None
        self._gemini_client = None

        if self.provider == "openai" and settings.OPENAI_API_KEY:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        elif self.provider == "anthropic" and settings.ANTHROPIC_API_KEY:
            from anthropic import AsyncAnthropic

            self._anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        elif self.provider == "gemini" and settings.GEMINI_API_KEY:
            # google-genai is Google's current SDK for Gemini 2.x and 3.x.
            # The client itself is sync-only; we call it via asyncio.to_thread
            # in _gemini_complete so we don't block the event loop.
            from google import genai

            self._gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        operation: str = "complete",
    ) -> str:
        provider = self.provider or "none"

        # Resolve the per-provider call once so retry overhead is just the call.
        if self.provider == "openai" and self._openai_client:
            call = self._openai_complete
        elif self.provider == "anthropic" and self._anthropic_client:
            call = self._anthropic_complete
        elif self.provider == "gemini" and self._gemini_client:
            call = self._gemini_complete
        else:
            llm_errors_total.labels(provider=provider, error_type="auth").inc()
            llm_requests_total.labels(
                provider=provider, operation=operation, status="failure"
            ).inc()
            raise LLMError(
                f"LLM provider '{self.provider}' is not configured. "
                "Set the appropriate API key."
            )

        last_exc: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            start = time.perf_counter()
            try:
                result = await call(system_prompt, user_prompt, max_tokens, temperature)
                llm_request_duration.labels(
                    provider=provider, operation=operation
                ).observe(time.perf_counter() - start)
                llm_requests_total.labels(
                    provider=provider, operation=operation, status="success"
                ).inc()
                return result
            except LLMError:
                # Empty response and other library-internal errors are not retryable.
                llm_requests_total.labels(
                    provider=provider, operation=operation, status="failure"
                ).inc()
                llm_errors_total.labels(
                    provider=provider, error_type="empty_response"
                ).inc()
                raise
            except Exception as exc:
                last_exc = exc
                error_type, retryable = _classify_error(exc)
                llm_errors_total.labels(provider=provider, error_type=error_type).inc()
                logger.warning(
                    "llm_request_error",
                    provider=provider,
                    operation=operation,
                    attempt=attempt,
                    error_type=error_type,
                    retryable=retryable,
                    error=str(exc),
                )

                if not retryable or attempt == MAX_ATTEMPTS:
                    llm_requests_total.labels(
                        provider=provider, operation=operation, status="failure"
                    ).inc()
                    raise LLMError(f"LLM request failed: {exc}") from exc

                llm_retries_total.labels(provider=provider, operation=operation).inc()
                backoff = BASE_BACKOFF_SEC * (2 ** (attempt - 1))
                jitter = random.uniform(0, backoff * 0.25)
                await asyncio.sleep(backoff + jitter)

        # Defensive — loop exits via return or raise above.
        raise LLMError(f"LLM request failed: {last_exc}")

    @staticmethod
    def _request_id_headers() -> dict[str, str]:
        rid = current_request_id()
        return {"X-Request-ID": rid} if rid else {}

    async def _openai_complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        assert self._openai_client is not None
        extra_headers = self._request_id_headers() or None
        response = await self._openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_headers=extra_headers,
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMError("LLM returned empty response")
        return content

    async def _anthropic_complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        assert self._anthropic_client is not None
        extra_headers = self._request_id_headers() or None
        response = await self._anthropic_client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_headers=extra_headers,
        )
        if not response.content:
            raise LLMError("LLM returned empty response")
        return response.content[0].text

    async def _gemini_complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        assert self._gemini_client is not None

        from google.genai import types

        # google-genai uses a single `contents` argument; the system prompt
        # is supplied via system_instruction in the generation config.
        config_kwargs: dict = {
            "system_instruction": system_prompt,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "response_mime_type": "application/json",
        }

        # Gemini 2.5+ models enable "thinking" by default, which silently
        # eats the entire max_output_tokens budget on internal reasoning
        # tokens before producing any visible text. For Wahb's workload
        # (translation, short summaries) we don't need it — turn it off so
        # the budget goes to actual output. Older models reject this field,
        # so only set it when the model name implies 2.5 or newer.
        model_name = (self.model or "").lower()
        if any(tag in model_name for tag in ("2.5", "3.", "3-", "flash-latest", "pro-latest")):
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=0
                )
            except AttributeError:
                # Older SDK without ThinkingConfig — skip silently.
                pass

        # Propagate X-Request-ID to Google API for cross-service tracing.
        rid = current_request_id()
        if rid:
            try:
                config_kwargs["http_options"] = types.HttpOptions(
                    headers={"X-Request-ID": rid}
                )
            except (AttributeError, TypeError):
                # Older SDK — skip header propagation silently.
                pass

        config = types.GenerateContentConfig(**config_kwargs)

        # The SDK is synchronous — offload to a thread so the FastAPI event
        # loop stays responsive under concurrent translate/summarize calls.
        response = await asyncio.to_thread(
            self._gemini_client.models.generate_content,
            model=self.model,
            contents=user_prompt,
            config=config,
        )

        text = getattr(response, "text", None)
        if not text:
            raise LLMError("LLM returned empty response")
        return text
