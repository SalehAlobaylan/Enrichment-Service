import asyncio
import random
import time
from collections.abc import Awaitable, Callable

from src.clients.llm_cache import LLMCache
from src.config import Settings
from src.middleware.error_handler import LLMError
from src.middleware.request_id import current_request_id
from src.utils.logging import get_logger
from src.utils.metrics import (
    llm_cache_hits_total,
    llm_cache_misses_total,
    llm_errors_total,
    llm_fallback_invocations_total,
    llm_request_duration,
    llm_requests_total,
    llm_retries_total,
)

# Caching is only safe for low-temperature ("deterministic-ish") prompts.
# Above this threshold, the model is intentionally sampling diversely and
# returning a cached response would defeat the operator's intent.
CACHE_TEMPERATURE_CEILING = 0.3

logger = get_logger(__name__)

# Retry policy — applied uniformly across providers per attempt.
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


# Type alias for a per-provider call: (system_prompt, user_prompt, max_tokens, temperature) -> text
_ProviderCall = Callable[[str, str, int, float], Awaitable[str]]


class LLMClient:
    """LLM client with multi-provider failover.

    Behavior:
      1. Run the primary provider through a retry loop (MAX_ATTEMPTS).
      2. If that exhausts on a retryable error, advance to the next provider
         in LLM_FALLBACK_PROVIDERS and try again.
      3. Non-retryable errors (auth/bad_request) from a provider DO trigger
         fallback — a fallback is more useful than letting auth issues nuke
         the request entirely.
      4. `LLMError` (empty response, missing config) is also fall-throughable
         so a one-off bad output from one provider doesn't kill the call.

    On overall failure, raises the last LLMError encountered.
    """

    def __init__(self, settings: Settings, cache: LLMCache | None = None):
        self.settings = settings
        self.primary_provider = (settings.LLM_PROVIDER or "").strip().lower()
        self._cache = cache if settings.LLM_CACHE_ENABLED else None
        self._clients: dict[str, object] = {}
        self._dispatch: dict[str, _ProviderCall] = {}

        # Instantiate ANY provider whose API key is set, regardless of which
        # is primary — so fallback paths have a ready client.
        if settings.OPENAI_API_KEY:
            from openai import AsyncOpenAI

            self._clients["openai"] = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self._dispatch["openai"] = self._openai_complete

        if settings.ANTHROPIC_API_KEY:
            from anthropic import AsyncAnthropic

            self._clients["anthropic"] = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            self._dispatch["anthropic"] = self._anthropic_complete

        if settings.GEMINI_API_KEY:
            from google import genai

            self._clients["gemini"] = genai.Client(api_key=settings.GEMINI_API_KEY)
            self._dispatch["gemini"] = self._gemini_complete

    @property
    def provider(self) -> str:
        """Backward-compat alias — some legacy callers read this."""
        return self.primary_provider

    @property
    def model(self) -> str:
        """Backward-compat alias for the primary model."""
        return self.settings.LLM_MODEL

    def _provider_chain(self) -> list[str]:
        """Primary first, then configured fallbacks. Empty list if no provider."""
        if not self.primary_provider or self.primary_provider in ("none", "disabled"):
            return []
        chain = [self.primary_provider, *self.settings.fallback_providers]
        # Drop any that don't have an instantiated client (e.g., missing API key).
        return [p for p in chain if p in self._dispatch]

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        operation: str = "complete",
    ) -> str:
        chain = self._provider_chain()
        if not chain:
            # No provider configured at all.
            llm_errors_total.labels(provider="none", error_type="auth").inc()
            llm_requests_total.labels(
                provider="none", operation=operation, status="failure"
            ).inc()
            raise LLMError(
                f"LLM provider '{self.primary_provider}' is not configured. "
                "Set the appropriate API key."
            )

        # Cache lookup — keyed on the PRIMARY provider/model so the cache is
        # stable regardless of which fallback ultimately serves a request.
        # Only cache when temperature is low (intent is deterministic) and
        # the cache is wired up.
        cache_eligible = (
            self._cache is not None and temperature <= CACHE_TEMPERATURE_CEILING
        )
        cache_key: str | None = None
        if cache_eligible:
            cache_key = LLMCache.make_key(
                provider=chain[0],
                model=self.settings.model_for(chain[0]),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            cached = await self._cache.get(cache_key)  # type: ignore[union-attr]
            if cached is not None:
                llm_cache_hits_total.labels(
                    provider=chain[0], operation=operation
                ).inc()
                logger.info(
                    "llm_cache_hit",
                    provider=chain[0],
                    operation=operation,
                )
                return cached
            llm_cache_misses_total.labels(
                provider=chain[0], operation=operation
            ).inc()

        last_error: LLMError | None = None
        winner: str | None = None
        for idx, provider in enumerate(chain):
            if idx > 0:
                llm_fallback_invocations_total.labels(
                    from_provider=chain[idx - 1],
                    to_provider=provider,
                    operation=operation,
                ).inc()
                logger.warning(
                    "llm_fallback_invoked",
                    from_provider=chain[idx - 1],
                    to_provider=provider,
                    operation=operation,
                )
            try:
                result = await self._attempt_provider(
                    provider, system_prompt, user_prompt, max_tokens, temperature, operation
                )
                winner = provider
                # Cache only when the PRIMARY provider succeeded — fallback
                # outputs come from a different model and shouldn't be served
                # as cached "primary" responses.
                if cache_eligible and cache_key and winner == chain[0]:
                    await self._cache.set(  # type: ignore[union-attr]
                        cache_key, result, self.settings.LLM_CACHE_TTL_SEC
                    )
                return result
            except LLMError as exc:
                last_error = exc
                continue

        # Every provider exhausted.
        assert last_error is not None
        raise last_error

    async def _attempt_provider(
        self,
        provider: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        operation: str,
    ) -> str:
        """Run one provider through the retry loop. Raises LLMError on exhaustion.

        This is the unit of work that fallback orchestration retries with a
        different provider. Per-provider metrics are emitted from here.
        """
        call = self._dispatch[provider]

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
            except LLMError as exc:
                # Empty response and other library-internal errors are not
                # retryable within this provider, but DO trigger fallback.
                llm_requests_total.labels(
                    provider=provider, operation=operation, status="failure"
                ).inc()
                llm_errors_total.labels(
                    provider=provider, error_type="empty_response"
                ).inc()
                raise LLMError(f"{provider}: {exc}") from exc
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
                    raise LLMError(f"{provider}: {exc}") from exc

                llm_retries_total.labels(provider=provider, operation=operation).inc()
                backoff = BASE_BACKOFF_SEC * (2 ** (attempt - 1))
                jitter = random.uniform(0, backoff * 0.25)
                await asyncio.sleep(backoff + jitter)

        # Defensive — loop exits via return or raise above.
        raise LLMError(f"{provider}: {last_exc}")

    @staticmethod
    def _request_id_headers() -> dict[str, str]:
        rid = current_request_id()
        return {"X-Request-ID": rid} if rid else {}

    async def _openai_complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        client = self._clients["openai"]
        model = self.settings.model_for("openai")
        extra_headers = self._request_id_headers() or None
        response = await client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
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
        client = self._clients["anthropic"]
        model = self.settings.model_for("anthropic")
        extra_headers = self._request_id_headers() or None
        response = await client.messages.create(  # type: ignore[attr-defined]
            model=model,
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
        client = self._clients["gemini"]
        model = self.settings.model_for("gemini")

        from google.genai import types

        config_kwargs: dict = {
            "system_instruction": system_prompt,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "response_mime_type": "application/json",
        }

        # Gemini 2.5+ models enable "thinking" by default, which silently
        # eats the entire max_output_tokens budget on internal reasoning
        # tokens before producing any visible text. Disable for Wahb's
        # short translation/summarization workload.
        model_name = (model or "").lower()
        if any(tag in model_name for tag in ("2.5", "3.", "3-", "flash-latest", "pro-latest")):
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=0
                )
            except AttributeError:
                pass

        # Propagate X-Request-ID to Google API for cross-service tracing.
        rid = current_request_id()
        if rid:
            try:
                config_kwargs["http_options"] = types.HttpOptions(
                    headers={"X-Request-ID": rid}
                )
            except (AttributeError, TypeError):
                pass

        config = types.GenerateContentConfig(**config_kwargs)

        # The SDK is synchronous — offload to a thread so the FastAPI event
        # loop stays responsive under concurrent translate/summarize calls.
        response = await asyncio.to_thread(
            client.models.generate_content,  # type: ignore[attr-defined]
            model=model,
            contents=user_prompt,
            config=config,
        )

        text = getattr(response, "text", None)
        if not text:
            raise LLMError("LLM returned empty response")
        return text
