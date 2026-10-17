import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.common.config import Settings
from src.common.middleware.error_handler import BudgetCappedError, LLMError
from src.llm.clients.llm import LLMClient
from src.llm.clients.spend_context import SpendAllowanceCache, SpendContext


@pytest.mark.asyncio
async def test_tenant_cache_cold_start_failure_and_stop_resume():
    cms = SimpleNamespace(
        get_ai_spend_allowance=AsyncMock(return_value={"allowed": False, "reason": "budget_capped"})
    )
    guard = SpendAllowanceCache(cms)
    a = SpendContext(tenant_id="a", trigger_source="autopilot")
    b = SpendContext(tenant_id="b")
    assert guard.check(a)["reason"] == "governor_unavailable"
    await guard._tasks[("a", "llm")]
    with pytest.raises(BudgetCappedError):
        await guard.authorize(a)
    assert guard.check(b)["allowed"]
    guard._entries[("a", "llm")] = (time.monotonic() - 121, {"allowed": False})
    cms.get_ai_spend_allowance.side_effect = RuntimeError("outage")
    assert guard.check(a)["ledger_health"] == "unavailable"
    await guard._tasks[("a", "llm")]
    await guard.authorize(a)
    await guard.close()


@pytest.mark.asyncio
async def test_exemption_registry_and_override_validation():
    cms = SimpleNamespace(get_ai_spend_allowance=AsyncMock(return_value={"allowed": True}))
    guard = SpendAllowanceCache(cms)
    guard._entries[("a", "llm")] = (time.monotonic(), {"allowed": False})
    with pytest.raises(BudgetCappedError):
        await guard.authorize(
            SpendContext(tenant_id="a", serving_exempt=True, trigger_source="forged")
        )
    await guard.authorize(
        SpendContext(tenant_id="a", serving_exempt=True, trigger_source="serving_news_summary")
    )
    ctx = SpendContext(tenant_id="a", trigger_source="manual", override_id="confirmation")
    await guard.authorize(ctx)
    cms.get_ai_spend_allowance.assert_awaited_once_with("a", "llm", "manual", "confirmation")
    await guard.close()


@pytest.mark.asyncio
async def test_paid_stop_cache_hit_empty_response_and_immutable_parallel_context():
    client = LLMClient(Settings(LLM_PROVIDER="openai", LLM_CACHE_ENABLED=False))
    provider = AsyncMock(return_value=("ok", {"input_tokens": 0, "output_tokens": 0}))
    client._dispatch = {"openai": provider}
    events = []
    client._spend_meter = SimpleNamespace(
        enqueue=events.append, unobservable_attempt=lambda _: None
    )
    client._allowance = SimpleNamespace(authorize=AsyncMock(side_effect=BudgetCappedError("cap")))
    with pytest.raises(BudgetCappedError):
        await client.complete("s", "u")
    provider.assert_not_awaited()
    client._cache = SimpleNamespace(
        get_entry=AsyncMock(return_value=("cached", {"input_tokens": 10, "output_tokens": 2}))
    )
    assert await client.complete("s", "u") == "cached"
    assert events[-1]["cached"]
    client._cache = None
    client._allowance = None
    a = client.for_context(
        SpendContext(
            tenant_id="a", trigger_source="studio", system_run_id="run-a",
            economics_attribution_ref="opaque-a",
        )
    )
    b = client.for_context(
        SpendContext(
            tenant_id="b", trigger_source="preferences", system_run_id="run-b",
            economics_attribution_ref="opaque-b",
        )
    )
    await asyncio.gather(a.complete("s", "a"), b.complete("s", "b"))
    assert {(e["tenant_id"], e["system_run_id"]) for e in events[-2:]} == {
        ("a", "run-a"),
        ("b", "run-b"),
    }
    assert {(e["tenant_id"], e["economics_attribution_ref"]) for e in events[-2:]} == {
        ("a", "opaque-a"), ("b", "opaque-b")
    }
    assert all(not e["estimated"] for e in events[-2:])
    provider.return_value = ("", {"input_tokens": 10, "output_tokens": 0})
    with pytest.raises(LLMError):
        await a.complete("s", "empty")
    assert events[-1]["units"]["input_tokens"] == 10
    provider.return_value = ("ok", {})
    await a.complete("s", "missing usage")
    assert events[-1]["estimated"] and events[-1]["units"]["input_tokens"] > 0
    await a.complete("s", "single translation request", operation="translate")
    assert events[-1]["units"]["operation_items"] == 1
    assert "operation_items" not in events[-2]["units"]


@pytest.mark.asyncio
async def test_unavailable_guard_is_durable_health_and_cached_admission_is_authoritative():
    counts = []
    cms = SimpleNamespace(get_ai_spend_allowance=AsyncMock(side_effect=RuntimeError("outage")))
    meter = SimpleNamespace(guard_unavailable=counts.append)
    guard = SpendAllowanceCache(cms, meter)
    await guard.authorize(SpendContext(tenant_id="a"))
    assert counts == ["a"]
    await guard._tasks[("a", "llm")]
    with pytest.raises(BudgetCappedError):
        await guard.authorize(
            SpendContext(tenant_id="a", admission={"allowed": False, "reason": "class_paused"})
        )
    await guard.close()


@pytest.mark.asyncio
async def test_optional_tagging_cap_has_explicit_skip_result():
    from src.llm.services.tagging import TaggingService

    llm = SimpleNamespace(complete=AsyncMock(side_effect=BudgetCappedError("cap")))
    result = await TaggingService(llm).extract("long article " * 100)
    assert result.skip_reason == "budget_capped" and result.tags == []


@pytest.mark.asyncio
async def test_cancellation_after_provider_start_records_uncertainty():
    client = LLMClient(Settings(LLM_PROVIDER="openai", LLM_CACHE_ENABLED=False))
    started = asyncio.Event()

    async def blocked(*args):
        started.set()
        await asyncio.Future()

    client._dispatch = {"openai": blocked}
    uncertainty = []
    client._spend_meter = SimpleNamespace(
        enqueue=lambda _: None, unobservable_attempt=uncertainty.append
    )
    task = asyncio.create_task(
        client.for_context(SpendContext(tenant_id="cancel-tenant")).complete("s", "u")
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert uncertainty == ["cancel-tenant"]


@pytest.mark.asyncio
async def test_named_item_denominator_counts_request_once_across_paid_fallback_responses():
    client = LLMClient(Settings(LLM_PROVIDER="openai", LLM_CACHE_ENABLED=False))
    client._provider_chain = lambda: ["openai", "gemini"]
    client._dispatch = {
        "openai": AsyncMock(return_value=("", {"input_tokens": 10, "output_tokens": 0})),
        "gemini": AsyncMock(return_value=("translated", {"input_tokens": 11, "output_tokens": 2})),
    }
    events = []
    client._spend_meter = SimpleNamespace(
        enqueue=events.append, unobservable_attempt=lambda _: None
    )
    assert await client.complete("s", "item", operation="translate") == "translated"
    assert len(events) == 2
    assert [e["units"]["operation_items"] for e in events] == [1, 0]
    assert sum(e["units"]["input_tokens"] for e in events) == 21
