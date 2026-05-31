"""Deep integration test for the DeepSeek provider via the REAL LLMClient.

Exercises the actual production code path (dispatch, retry loop, fallback
orchestration, error classification, metrics) against the live DeepSeek API.
No models loaded, no FastAPI app — just the LLM client.

Run:  .venv/bin/python scripts/deepseek_deeptest.py
"""

import asyncio
import os
import sys

# Load LLM keys from .env.local at repo root without sourcing (avoids `&` issues).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_PATH = os.path.join(REPO_ROOT, ".env.local")


def load_keys() -> dict[str, str]:
    keys: dict[str, str] = {}
    wanted = {"DEEPSEEK_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
    if not os.path.exists(ENV_PATH):
        return keys
    with open(ENV_PATH, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            if k in wanted:
                keys[k] = v.strip().strip('"').strip("'")
    return keys


KEYS = load_keys()

from src.common.config import Settings  # noqa: E402
from src.common.middleware.error_handler import LLMError  # noqa: E402
from src.llm.clients.llm import LLMClient  # noqa: E402


def line(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def make_settings(**overrides) -> Settings:
    base = dict(
        LLM_CACHE_ENABLED=False,  # isolate provider behavior from Redis
        DEEPSEEK_API_KEY=KEYS.get("DEEPSEEK_API_KEY", ""),
        GEMINI_API_KEY=KEYS.get("GEMINI_API_KEY", ""),
        OPENAI_API_KEY=KEYS.get("OPENAI_API_KEY", ""),
        ANTHROPIC_API_KEY=KEYS.get("ANTHROPIC_API_KEY", ""),
    )
    base.update(overrides)
    return Settings(**base)


async def test_1_direct_deepseek() -> bool:
    line("TEST 1 — Direct DeepSeek through real dispatch (_attempt_provider)")
    s = make_settings(LLM_PROVIDER="deepseek", LLM_FALLBACK_PROVIDERS="")
    client = LLMClient(s)
    print(f"chain={client._provider_chain()}  model={s.model_for('deepseek')}")
    try:
        out = await client._attempt_provider(
            "deepseek",
            "You are a terse assistant. Reply with one word only.",
            "Say the word: WORKING",
            max_tokens=8,
            temperature=0.0,
            operation="smoke",
        )
        print(f"RESULT: {out!r}")
        return "WORKING" in out.upper()
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


async def test_2_fallback_chain() -> bool:
    line("TEST 2 — Fallback: bad Gemini key -> DeepSeek wins (.complete)")
    # Primary gemini with a deliberately invalid key; deepseek as fallback.
    s = make_settings(
        LLM_PROVIDER="gemini",
        GEMINI_API_KEY="AIza-INVALID-KEY-FOR-FALLBACK-TEST",
        LLM_FALLBACK_PROVIDERS="deepseek",
    )
    client = LLMClient(s)
    print(f"chain={client._provider_chain()}")
    try:
        out = await client.complete(
            "You are a terse assistant. Reply with one word only.",
            "Reply with the word: FALLBACK",
            max_tokens=8,
            temperature=0.0,
            operation="fallback_test",
        )
        print(f"RESULT (served by DeepSeek after Gemini failed): {out!r}")
        return "FALLBACK" in out.upper()
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


async def test_3_arabic_quality() -> bool:
    line("TEST 3 — Arabic quality on real translate / summarize / tag tasks")
    s = make_settings(LLM_PROVIDER="deepseek", LLM_FALLBACK_PROVIDERS="")
    client = LLMClient(s)

    article = (
        "أعلنت الحكومة السعودية اليوم عن إطلاق مبادرة وطنية جديدة لمكافحة التغير "
        "المناخي تهدف إلى زراعة عشرة مليارات شجرة بحلول عام 2030 ضمن مشروع "
        "السعودية الخضراء. وقال ولي العهد إن المبادرة ستسهم في خفض الانبعاثات "
        "الكربونية وتحسين جودة الحياة في المدن الكبرى مثل الرياض وجدة."
    )

    ok = True

    print("\n[3a] Translate AR -> EN")
    try:
        tr = await client.complete(
            "You are a professional translator. Translate the user's Arabic text "
            "into clear, natural English. Output only the translation.",
            article,
            max_tokens=400,
            temperature=0.0,
            operation="translate",
        )
        print(tr.strip())
        ok = ok and len(tr.strip()) > 40
    except Exception as exc:
        print(f"FAILED: {exc}")
        ok = False

    print("\n[3b] Summarize in Arabic (1 sentence)")
    try:
        sm = await client.complete(
            "أنت مساعد تحرير. لخّص النص في جملة واحدة بالعربية الفصحى.",
            article,
            max_tokens=150,
            temperature=0.0,
            operation="summarize",
        )
        print(sm.strip())
        ok = ok and len(sm.strip()) > 10
    except Exception as exc:
        print(f"FAILED: {exc}")
        ok = False

    print("\n[3c] Topic tags (JSON array)")
    try:
        tg = await client.complete(
            "Extract 3-5 short lowercase topic tags from the text. "
            'Respond ONLY as a JSON array of strings, e.g. ["a","b"].',
            article,
            max_tokens=80,
            temperature=0.0,
            operation="tagging",
        )
        print(tg.strip())
        ok = ok and "[" in tg
    except Exception as exc:
        print(f"FAILED: {exc}")
        ok = False

    return ok


async def test_4_resilience_bad_key() -> bool:
    line("TEST 4 — Resilience: invalid DeepSeek key -> auth error, no hang")
    s = make_settings(
        LLM_PROVIDER="deepseek",
        DEEPSEEK_API_KEY="sk-INVALID-KEY-000000000000000000",
        LLM_FALLBACK_PROVIDERS="",
    )
    client = LLMClient(s)
    import time

    start = time.perf_counter()
    try:
        await client.complete(
            "x", "y", max_tokens=8, temperature=0.0, operation="resilience"
        )
        print("UNEXPECTED: call succeeded with an invalid key")
        return False
    except LLMError as exc:
        elapsed = time.perf_counter() - start
        print(f"Correctly raised LLMError in {elapsed:.1f}s: {exc}")
        # Auth errors are non-retryable -> should fail fast (< ~5s), not 3x backoff.
        return elapsed < 10
    except Exception as exc:
        print(f"Raised non-LLMError: {type(exc).__name__}: {exc}")
        return False


async def main() -> None:
    if not KEYS.get("DEEPSEEK_API_KEY"):
        print("No DEEPSEEK_API_KEY found in .env.local — aborting.")
        sys.exit(1)

    results = {}
    results["1. direct deepseek"] = await test_1_direct_deepseek()
    results["2. fallback chain"] = await test_2_fallback_chain()
    results["3. arabic quality"] = await test_3_arabic_quality()
    results["4. resilience"] = await test_4_resilience_bad_key()

    line("SUMMARY")
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print()
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    asyncio.run(main())
