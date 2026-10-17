import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.common.config import Settings
from src.common.middleware.error_handler import LLMError
from src.llm.clients import llm as llm_module
from src.llm.clients.llm import LLMClient
from src.llm.clients.llm_cache import LLMCache


def _client(*, fallback: str = "") -> LLMClient:
    client = LLMClient(
        Settings(
            LLM_PROVIDER="openai",
            LLM_FALLBACK_PROVIDERS=fallback,
            LLM_CACHE_ENABLED=False,
        )
    )
    # SDK construction depends on API keys; these direct tests use only the
    # dispatch seam and never make a network call.
    return client


@pytest.mark.asyncio
async def test_invalid_cached_output_is_regenerated_and_only_valid_output_cached() -> None:
    client = _client()
    cache = AsyncMock()
    cache.get_entry.return_value = ("malformed", {})
    client._cache = cache
    async def primary(*_args):
        return "valid", {"output_tokens": 10}
    client._dispatch = {"openai": primary}
    def validate(text):
        if text != "valid":
            raise LLMError("invalid output")
    result = await client.complete("system", "user", validate_output=validate)
    assert result == "valid"
    cache.set_entry.assert_awaited_once()
    assert cache.set_entry.await_args.args[1] == "valid"


@pytest.mark.asyncio
async def test_invalid_fresh_output_is_not_cached_but_is_metered() -> None:
    client = _client()
    cache = AsyncMock()
    cache.get_entry.return_value = None
    client._cache = cache
    meter = SimpleNamespace(enqueue=lambda event: events.append(event))
    events = []
    client._spend_meter = meter
    async def primary(*_args):
        return "malformed", {"output_tokens": 10}
    client._dispatch = {"openai": primary}
    def validate(_text):
        raise LLMError("invalid output")
    with pytest.raises(LLMError, match="invalid output"):
        await client.complete("system", "user", validate_output=validate)
    cache.set_entry.assert_not_awaited()
    assert len(events) == 1 and events[0]["units"]["output_tokens"] == 10


@pytest.mark.asyncio
async def test_transient_primary_failure_uses_fallback_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(fallback="deepseek")
    calls: list[str] = []

    class ServerError(Exception):
        status_code = 503

    async def primary(*_args: object) -> tuple[str, dict[str, int]]:
        calls.append("openai")
        raise ServerError("unavailable")

    async def fallback(*_args: object) -> tuple[str, dict[str, int]]:
        calls.append("deepseek")
        return "fallback answer", {"input_tokens": 2, "output_tokens": 1}

    monkeypatch.setattr(llm_module, "MAX_ATTEMPTS", 1)
    client._dispatch = {"openai": primary, "deepseek": fallback}

    assert await client.complete("system", "user") == "fallback answer"
    assert calls == ["openai", "deepseek"]


@pytest.mark.asyncio
async def test_deadline_prevents_fallback_and_late_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(fallback="deepseek")
    calls: list[str] = []

    async def slow_primary(*_args: object) -> tuple[str, dict[str, int]]:
        calls.append("openai")
        await asyncio.sleep(0.05)
        return "late", {}

    async def fallback(*_args: object) -> tuple[str, dict[str, int]]:
        calls.append("deepseek")
        return "unexpected", {}

    monkeypatch.setattr(llm_module, "REQUEST_DEADLINE_SEC", 0.01)
    monkeypatch.setattr(llm_module, "MIN_ATTEMPT_BUDGET_SEC", 0.001)
    client._dispatch = {"openai": slow_primary, "deepseek": fallback}

    with pytest.raises(LLMError, match="deadline exceeded"):
        await client.complete("system", "user")
    assert calls == ["openai"]


@pytest.mark.asyncio
async def test_cancellation_propagates_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(fallback="deepseek")
    started = asyncio.Event()
    fallback_called = False

    async def slow_primary(*_args: object) -> tuple[str, dict[str, int]]:
        started.set()
        await asyncio.sleep(10)
        return "late", {}

    async def fallback(*_args: object) -> tuple[str, dict[str, int]]:
        nonlocal fallback_called
        fallback_called = True
        return "unexpected", {}

    client._dispatch = {"openai": slow_primary, "deepseek": fallback}
    task = asyncio.create_task(client.complete("system", "user"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fallback_called is False


@pytest.mark.asyncio
async def test_close_closes_every_initialized_client() -> None:
    client = _client()

    class FakeClient:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    first, second = FakeClient(), FakeClient()
    client._clients = {"one": first, "two": second}
    await client.close()
    assert first.closed is True
    assert second.closed is True


@pytest.mark.asyncio
async def test_provider_adapters_use_their_own_models_and_usage() -> None:
    calls: dict[str, str] = {}

    class OpenAICompletions:
        def __init__(self, provider: str) -> None:
            self.provider = provider

        async def create(self, **kwargs: object) -> object:
            calls[self.provider] = str(kwargs["model"])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="openai"))],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            )

    class AnthropicMessages:
        async def create(self, **kwargs: object) -> object:
            calls["anthropic"] = str(kwargs["model"])
            return SimpleNamespace(
                content=[SimpleNamespace(text="anthropic")],
                usage=SimpleNamespace(input_tokens=4, output_tokens=1),
            )

    settings = Settings(
        LLM_PROVIDER="gemini",
        LLM_MODEL="gemini-primary-only",
        LLM_OPENAI_MODEL="openai-override",
        LLM_ANTHROPIC_MODEL="anthropic-override",
        LLM_DEEPSEEK_MODEL="deepseek-override",
        LLM_CACHE_ENABLED=False,
    )
    client = LLMClient(settings)
    client._clients = {
        "openai": SimpleNamespace(
            chat=SimpleNamespace(completions=OpenAICompletions("openai"))
        ),
        "deepseek": SimpleNamespace(
            chat=SimpleNamespace(completions=OpenAICompletions("deepseek"))
        ),
        "anthropic": SimpleNamespace(messages=AnthropicMessages()),
    }

    assert await client._openai_complete("s", "u", 10, 0.1) == (
        "openai",
        {"input_tokens": 3, "output_tokens": 2},
    )
    assert await client._deepseek_complete("s", "u", 10, 0.1) == (
        "openai",
        {"input_tokens": 3, "output_tokens": 2, "input_cache_hit_tokens": 0},
    )
    assert await client._anthropic_complete("s", "u", 10, 0.1) == (
        "anthropic",
        {"input_tokens": 4, "output_tokens": 1},
    )
    assert calls == {
        "openai": "openai-override",
        "anthropic": "anthropic-override",
        "deepseek": "deepseek-override",
    }


@pytest.mark.asyncio
async def test_gemini_adapter_extracts_usage_with_its_own_model() -> None:
    model_seen: str | None = None

    class GeminiModels:
        def generate_content(self, **kwargs: object) -> object:
            nonlocal model_seen
            model_seen = str(kwargs["model"])
            return SimpleNamespace(
                text="gemini",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=5,
                    candidates_token_count=2,
                    thoughts_token_count=1,
                ),
            )

    client = LLMClient(
        Settings(
            LLM_PROVIDER="openai",
            LLM_MODEL="openai-primary-only",
            LLM_GEMINI_MODEL="gemini-override",
            LLM_CACHE_ENABLED=False,
        )
    )
    client._clients = {"gemini": SimpleNamespace(models=GeminiModels())}

    assert await client._gemini_complete("s", "u", 10, 0.1) == (
        "gemini",
        {"input_tokens": 5, "output_tokens": 2, "thoughts_tokens": 1},
    )
    assert model_seen == "gemini-override"


class _MemoryCache:
    def __init__(self) -> None:
        self.entries: dict[str, tuple[str, dict[str, int]]] = {}

    async def get_entry(self, key: str) -> tuple[str, dict[str, int]] | None:
        return self.entries.get(key)

    async def set_entry(
        self, key: str, value: str, usage: dict[str, int], _ttl: int
    ) -> None:
        self.entries[key] = (value, usage)


@pytest.mark.asyncio
async def test_primary_result_is_cached_but_fallback_result_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _MemoryCache()
    client = LLMClient(
        Settings(
            LLM_PROVIDER="openai",
            LLM_FALLBACK_PROVIDERS="deepseek",
            LLM_CACHE_ENABLED=True,
        ),
        cache=cache,  # type: ignore[arg-type]
    )
    primary_calls = 0
    fallback_calls = 0

    async def primary(*_args: object) -> tuple[str, dict[str, int]]:
        nonlocal primary_calls
        primary_calls += 1
        if primary_calls == 1:
            return "primary", {"input_tokens": 1, "output_tokens": 1}
        raise type("ServerError", (Exception,), {"status_code": 503})("down")

    async def fallback(*_args: object) -> tuple[str, dict[str, int]]:
        nonlocal fallback_calls
        fallback_calls += 1
        return "fallback", {"input_tokens": 1, "output_tokens": 1}

    client._dispatch = {"openai": primary, "deepseek": fallback}
    monkeypatch.setattr(llm_module, "MAX_ATTEMPTS", 1)
    assert await client.complete("s", "first") == "primary"
    assert await client.complete("s", "first") == "primary"
    assert primary_calls == 1

    assert await client.complete("s", "second") == "fallback"
    assert await client.complete("s", "second") == "fallback"
    assert primary_calls == 3
    assert fallback_calls == 2


def test_cache_key_includes_version_provider_and_model() -> None:
    base = LLMCache.make_key("openai", "model-a", "s", "u", 10, 0.1)
    assert ":v1:" in base
    assert base != LLMCache.make_key("deepseek", "model-a", "s", "u", 10, 0.1)
    assert base != LLMCache.make_key("openai", "model-b", "s", "u", 10, 0.1)

@pytest.mark.asyncio
@pytest.mark.parametrize("hits", [123, None])
async def test_deepseek_retains_provider_cached_input_usage(hits) -> None:
    client = _client()
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=50)
    if hits is not None:
        usage.prompt_cache_hit_tokens = hits
    create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        usage=usage,
    ))
    client._clients["deepseek"] = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    _, measured = await client._deepseek_complete("system", "user", 100, 0)
    assert measured == {"input_tokens": 1000, "output_tokens": 50,
                        "input_cache_hit_tokens": hits or 0}
    events = []
    client._spend_meter = SimpleNamespace(enqueue=events.append)
    client._emit_spend(provider="deepseek", model="deepseek-chat", operation="test",
                       usage=measured, cached=False, trigger_source="test",
                       system_run_id=None, tenant_id="test")
    assert events[0]["units"]["input_cache_hit_tokens"] == (hits or 0)
    assert events[0]["cached"] is False
