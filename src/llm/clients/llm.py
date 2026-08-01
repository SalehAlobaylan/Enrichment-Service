import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from src.common.config import Settings
from src.common.middleware.error_handler import LLMError
from src.common.middleware.request_id import current_request_id
from src.common.utils.logging import get_logger
from src.common.utils.metrics import (
    llm_attempts_total,
    llm_cache_hits_total,
    llm_cache_misses_total,
    llm_deadline_exhaustions_total,
    llm_errors_total,
    llm_fallback_invocations_total,
    llm_request_duration,
    llm_requests_total,
    llm_retries_total,
)
from src.common.workload_admission import WorkloadAdmission, WorkloadExecutors
from src.llm.clients.llm_cache import LLMCache
from src.llm.clients.spend_meter import SpendMeter

if TYPE_CHECKING:
    from src.common.clients.cms import CMSClient

# Caching is only safe for low-temperature ("deterministic-ish") prompts.
# Above this threshold, the model is intentionally sampling diversely and
# returning a cached response would defeat the operator's intent.
CACHE_TEMPERATURE_CEILING = 0.3

logger = get_logger(__name__)

# Retry policy — applied uniformly across providers per attempt.
MAX_ATTEMPTS = 3
BASE_BACKOFF_SEC = 1.0  # 1s, 2s, 4s with jitter
# This is deliberately a code policy, rather than another operator tuning
# knob. Every provider attempt, retry sleep, and fallback shares this one
# budget, keeping a slow outage from multiplying request latency or spend.
REQUEST_DEADLINE_SEC = 30.0
MIN_ATTEMPT_BUDGET_SEC = 0.25


def _retry_after_seconds(exc: Exception) -> float | None:
    """Read Retry-After without exposing provider-specific error classes."""
    retry_after = getattr(exc, "retry_after", None)
    if isinstance(retry_after, (int, float)) and retry_after >= 0:
        return float(retry_after)

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, IndexError, OverflowError):
            return None


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
_ProviderCall = Callable[[str, str, int, float], Awaitable[tuple[str, dict[str, int]]]]


@dataclass(frozen=True)
class LLMCompletion:
    """Completion text plus service-observed provider provenance."""

    text: str
    provider: str
    model: str
    fallback_used: bool
    cached: bool


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

    def __init__(
        self,
        settings: Settings,
        cache: LLMCache | None = None,
        cms_client: "CMSClient | None" = None,
        admission: WorkloadAdmission | None = None,
        spend_meter: SpendMeter | None = None,
        executors: WorkloadExecutors | None = None,
    ) -> None:
        self.settings = settings
        self.primary_provider = (settings.LLM_PROVIDER or "").strip().lower()
        self._cache = cache if settings.LLM_CACHE_ENABLED else None
        self._cms_client = cms_client
        self._admission = admission
        self._spend_meter = spend_meter
        self._executors = executors
        self._clients: dict[str, object] = {}
        self._dispatch: dict[str, _ProviderCall] = {}

        # Instantiate ANY provider whose API key is set, regardless of which
        # is primary — so fallback paths have a ready client.
        if settings.OPENAI_API_KEY:
            from openai import AsyncOpenAI

            self._clients["openai"] = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=REQUEST_DEADLINE_SEC,
                max_retries=0,
            )
            self._dispatch["openai"] = self._openai_complete

        if settings.ANTHROPIC_API_KEY:
            from anthropic import AsyncAnthropic

            self._clients["anthropic"] = AsyncAnthropic(
                api_key=settings.ANTHROPIC_API_KEY,
                timeout=REQUEST_DEADLINE_SEC,
                max_retries=0,
            )
            self._dispatch["anthropic"] = self._anthropic_complete

        if settings.GEMINI_API_KEY:
            import google.genai as genai

            self._clients["gemini"] = genai.Client(api_key=settings.GEMINI_API_KEY)
            self._dispatch["gemini"] = self._gemini_complete

        if settings.DEEPSEEK_API_KEY:
            # DeepSeek is OpenAI-compatible — same chat.completions API, same
            # request/response shape. We instantiate a separate AsyncOpenAI
            # client pointed at DeepSeek's base URL so the SDK code path is
            # identical to OpenAI's; the only difference is where requests
            # land. See https://api-docs.deepseek.com/ for the contract.
            from openai import AsyncOpenAI

            self._clients["deepseek"] = AsyncOpenAI(
                api_key=settings.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com/v1",
                timeout=REQUEST_DEADLINE_SEC,
                max_retries=0,
            )
            self._dispatch["deepseek"] = self._deepseek_complete

    @property
    def provider(self) -> str:
        """Backward-compat alias — some legacy callers read this."""
        return self.primary_provider

    @property
    def model(self) -> str:
        """Backward-compat alias for the primary model."""
        return self.settings.LLM_MODEL

    async def close(self) -> None:
        """Close SDK transports without letting shutdown failures block exit."""
        for client in self._clients.values():
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close is None:
                continue
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.warning("llm_client_close_failed", client_type=type(client).__name__)

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
        trigger_source: str = "unknown",
        system_run_id: str | None = None,
        tenant_id: str = "default",
        allow_cache: bool = True,
    ) -> str:
        result = await self.complete_with_provenance(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            operation=operation,
            trigger_source=trigger_source,
            system_run_id=system_run_id,
            tenant_id=tenant_id,
            allow_cache=allow_cache,
        )
        return result.text

    async def complete_with_provenance(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        operation: str = "complete",
        trigger_source: str = "unknown",
        system_run_id: str | None = None,
        tenant_id: str = "default",
        allow_cache: bool = True,
    ) -> LLMCompletion:
        chain = self._provider_chain()
        if not chain:
            # No provider configured at all.
            llm_errors_total.labels(provider="none", error_type="auth").inc()
            llm_requests_total.labels(provider="none", operation=operation, status="failure").inc()
            raise LLMError(
                f"LLM provider '{self.primary_provider}' is not configured. "
                "Set the appropriate API key."
            )

        # Cache lookup — keyed on the PRIMARY provider/model so the cache is
        # stable regardless of which fallback ultimately serves a request.
        # Only cache when temperature is low (intent is deterministic) and
        # the cache is wired up.
        cache_eligible = (
            allow_cache and self._cache is not None and temperature <= CACHE_TEMPERATURE_CEILING
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
            cached_entry = await self._cache.get_entry(cache_key)  # type: ignore[union-attr]
            if cached_entry is not None:
                cached, cached_usage = cached_entry
                llm_cache_hits_total.labels(provider=chain[0], operation=operation).inc()
                logger.info(
                    "llm_cache_hit",
                    provider=chain[0],
                    operation=operation,
                )
                self._emit_spend(
                    provider=chain[0],
                    model=self.settings.model_for(chain[0]),
                    operation=operation,
                    usage=cached_usage or {},
                    cached=True,
                    trigger_source=trigger_source,
                    system_run_id=system_run_id,
                    tenant_id=tenant_id,
                )
                return LLMCompletion(
                    text=cached,
                    provider=chain[0],
                    model=self.settings.model_for(chain[0]),
                    fallback_used=False,
                    cached=True,
                )
            llm_cache_misses_total.labels(provider=chain[0], operation=operation).inc()

        last_error: LLMError | None = None
        winner: str | None = None
        deadline = asyncio.get_running_loop().time() + REQUEST_DEADLINE_SEC
        for idx, provider in enumerate(chain):
            if self._remaining_budget(deadline) < MIN_ATTEMPT_BUDGET_SEC:
                last_error = self._deadline_error(operation)
                break
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
                result, usage = await self._attempt_provider(
                    provider,
                    system_prompt,
                    user_prompt,
                    max_tokens,
                    temperature,
                    operation,
                    deadline,
                )
                winner = provider
                # Cache only when the PRIMARY provider succeeded — fallback
                # outputs come from a different model and shouldn't be served
                # as cached "primary" responses.
                if cache_eligible and cache_key and winner == chain[0]:
                    await self._cache.set_entry(  # type: ignore[union-attr]
                        cache_key, result, usage, self.settings.LLM_CACHE_TTL_SEC
                    )
                self._emit_spend(
                    provider=provider,
                    model=self.settings.model_for(provider),
                    operation=operation,
                    usage=usage,
                    cached=False,
                    trigger_source=trigger_source,
                    system_run_id=system_run_id,
                    tenant_id=tenant_id,
                )
                return LLMCompletion(
                    text=result,
                    provider=provider,
                    model=self.settings.model_for(provider),
                    fallback_used=idx > 0,
                    cached=False,
                )
            except LLMError as exc:
                last_error = exc
                continue

        # Every provider exhausted.
        assert last_error is not None
        raise last_error

    @staticmethod
    def _remaining_budget(deadline: float) -> float:
        return deadline - asyncio.get_running_loop().time()

    @staticmethod
    def _deadline_error(operation: str) -> LLMError:
        llm_deadline_exhaustions_total.labels(operation=operation).inc()
        return LLMError("LLM request deadline exceeded")

    async def _attempt_provider(
        self,
        provider: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        operation: str,
        deadline: float,
    ) -> tuple[str, dict[str, int]]:
        """Run one provider through the retry loop. Raises LLMError on exhaustion.

        This is the unit of work that fallback orchestration retries with a
        different provider. Per-provider metrics are emitted from here.
        """
        call = self._dispatch[provider]

        last_exc: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            remaining = self._remaining_budget(deadline)
            if remaining < MIN_ATTEMPT_BUDGET_SEC:
                raise self._deadline_error(operation)
            start = time.perf_counter()
            try:
                if self._admission is None:
                    async with asyncio.timeout(remaining):
                        result = await call(system_prompt, user_prompt, max_tokens, temperature)
                else:
                    async with self._admission.acquire("llm_sync"):
                        async with asyncio.timeout(remaining):
                            result = await call(system_prompt, user_prompt, max_tokens, temperature)
                llm_attempts_total.labels(
                    provider=provider, operation=operation, status="success"
                ).inc()
                llm_request_duration.labels(provider=provider, operation=operation).observe(
                    time.perf_counter() - start
                )
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
                llm_errors_total.labels(provider=provider, error_type="empty_response").inc()
                llm_attempts_total.labels(
                    provider=provider, operation=operation, status="failure"
                ).inc()
                raise LLMError(f"{provider}: {exc}") from exc
            except TimeoutError as exc:
                last_exc = exc
                llm_errors_total.labels(provider=provider, error_type="timeout").inc()
                llm_attempts_total.labels(
                    provider=provider, operation=operation, status="timeout"
                ).inc()
                if self._remaining_budget(deadline) < MIN_ATTEMPT_BUDGET_SEC:
                    raise self._deadline_error(operation) from exc
                # A provider attempt timeout is transient, but remains subject
                # to the same global budget before another provider is tried.
                if attempt == MAX_ATTEMPTS:
                    llm_requests_total.labels(
                        provider=provider, operation=operation, status="failure"
                    ).inc()
                    raise LLMError(f"{provider}: request timed out") from exc
            except Exception as exc:
                last_exc = exc
                error_type, retryable = _classify_error(exc)
                llm_errors_total.labels(provider=provider, error_type=error_type).inc()
                llm_attempts_total.labels(
                    provider=provider, operation=operation, status="failure"
                ).inc()
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

            # Retry timing intentionally lives outside the exception branches:
            # both SDK timeouts and retryable transport/status errors use it.
            if attempt < MAX_ATTEMPTS:
                llm_retries_total.labels(provider=provider, operation=operation).inc()
                backoff = _retry_after_seconds(last_exc) if last_exc else None
                if backoff is None:
                    base = BASE_BACKOFF_SEC * (2 ** (attempt - 1))
                    backoff = base + random.uniform(0, base * 0.25)
                remaining_after_backoff = self._remaining_budget(deadline) - backoff
                if remaining_after_backoff < MIN_ATTEMPT_BUDGET_SEC:
                    raise self._deadline_error(operation)
                await asyncio.sleep(backoff)

        # Defensive — loop exits via return or raise above.
        raise LLMError(f"{provider}: {last_exc}")

    @staticmethod
    def _request_id_headers() -> dict[str, str]:
        rid = current_request_id()
        return {"X-Request-ID": rid} if rid else {}

    def _emit_spend(
        self,
        *,
        provider: str,
        model: str,
        operation: str,
        usage: dict[str, int],
        cached: bool,
        trigger_source: str,
        system_run_id: str | None,
        tenant_id: str,
    ) -> None:
        """Schedule metering without coupling LLM availability to CMS."""
        if self._spend_meter is None:
            return
        event: dict[str, Any] = {
            "event_id": str(uuid4()),
            "spend_class": "llm",
            "operation": operation,
            "provider": provider,
            "model": model,
            "units": usage,
            "cached": cached,
            "estimated": not bool(usage),
            "avoided_cost_estimated": cached and not bool(usage),
            "trigger_source": trigger_source,
            "system_run_id": system_run_id or "",
            "tenant_id": tenant_id,
            "source_service": "enrichment-service",
        }
        self._spend_meter.enqueue(event)

    async def _openai_complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, dict[str, int]]:
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
        usage = response.usage
        return content, {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }

    async def _deepseek_complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, dict[str, int]]:
        # DeepSeek speaks the OpenAI chat-completions wire format, so the
        # call is literally identical to _openai_complete — only the client
        # instance (pointed at api.deepseek.com) differs. Kept as a separate
        # method so the dispatch table + metrics + classification still see
        # "deepseek" as a distinct provider name.
        client = self._clients["deepseek"]
        model = self.settings.model_for("deepseek")
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
        usage = response.usage
        return content, {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }

    async def _anthropic_complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, dict[str, int]]:
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
        usage = response.usage
        return response.content[0].text, {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }

    async def _gemini_complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, dict[str, int]]:
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
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except AttributeError:
                pass

        # Propagate X-Request-ID to Google API for cross-service tracing.
        rid = current_request_id()
        if rid:
            try:
                config_kwargs["http_options"] = types.HttpOptions(headers={"X-Request-ID": rid})
            except (AttributeError, TypeError):
                pass

        config = types.GenerateContentConfig(**config_kwargs)

        # The SDK is synchronous — offload to a thread so the FastAPI event
        # loop stays responsive under concurrent translate/summarize calls.
        if self._executors is not None:
            response = await self._executors.run(
                "llm_sync",
                client.models.generate_content,  # type: ignore[attr-defined]
                model=model,
                contents=user_prompt,
                config=config,
            )
        else:
            response = await asyncio.to_thread(
                client.models.generate_content,  # type: ignore[attr-defined]
                model=model,
                contents=user_prompt,
                config=config,
            )

        text = getattr(response, "text", None)
        if not text:
            raise LLMError("LLM returned empty response")
        usage = getattr(response, "usage_metadata", None)
        return text, {
            "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
            "thoughts_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0),
        }
