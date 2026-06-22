import asyncio
import json

from src.common.utils.logging import get_logger
from src.llm.clients.llm import LLMClient
from src.llm.schemas.classify import AccountClassResult, AccountToClassify

logger = get_logger(__name__)

VALID = {"official", "news", "person", "other"}

SYSTEM_PROMPT = """
You classify a social-media account into EXACTLY ONE category, using its name and
bio. The account may be in Arabic or English — judge by MEANING, not keywords.

Categories:
- official: a government / ministry / state agency / regulator / ruler or royal /
  official public body / embassy / institution's official account.
- news: a news outlet, news agency, newspaper, TV or radio channel, or a news
  program/show.
- person: an individual — journalist, anchor, columnist, commentator, writer,
  analyst, or public figure.
- other: anything else (company, brand, sports club, fan account, generic).

Return ONLY a JSON object: {"class":"official"} (or news / person / other).
""".strip()

MAX_BIO_CHARS = 400
CONCURRENCY = 5


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    return text.strip()


class AccountClassificationService:
    """Deterministic-fallback LLM classifier for ambiguous source accounts.

    One `complete()` per account (temperature 0 → content-addressable cache, so a
    handle+bio classified once is served from Redis on every later build).
    Bounded concurrency; any failure/garbage degrades to "other".
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def classify(self, accounts: list[AccountToClassify]) -> list[AccountClassResult]:
        sem = asyncio.Semaphore(CONCURRENCY)

        async def one(acc: AccountToClassify) -> AccountClassResult:
            async with sem:
                cls = await self._classify_one(acc.handle, acc.name, acc.bio)
            return AccountClassResult(handle=acc.handle, source_class=cls)

        return list(await asyncio.gather(*[one(a) for a in accounts]))

    async def _classify_one(self, handle: str, name: str, bio: str) -> str:
        user_prompt = (
            f"Handle: @{handle}\n"
            f"Name: {(name or '').strip()}\n"
            f"Bio: {(bio or '').strip()[:MAX_BIO_CHARS]}"
        )
        try:
            raw = await self.llm.complete(
                SYSTEM_PROMPT,
                user_prompt,
                max_tokens=12,
                temperature=0.0,
                operation="classify_source",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("classify_failed", handle=handle, error=str(exc))
            return "other"
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> str:
        text = _strip_fences(raw).strip()
        cls = ""
        try:
            cls = str(json.loads(text).get("class", "")).strip().lower()
        except Exception:
            cls = text.strip('"').strip().lower()
        if cls in VALID:
            return cls
        # Salvage: a valid token anywhere in the response.
        for v in VALID:
            if v in cls:
                return v
        return "other"
