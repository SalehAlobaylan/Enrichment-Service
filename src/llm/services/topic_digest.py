import json

from src.common.utils.logging import get_logger
from src.llm.clients.llm import LLMClient
from src.llm.schemas.topic_digest import TopicDigestResponse

logger = get_logger(__name__)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1 :] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    return text.strip()


# The finite news category taxonomy. The model picks exactly one; anything not
# in this set (or missing) is normalized to "general", which the UI renders as
# no chip. Keep in sync with the CMS/UI slug list.
CATEGORIES = frozenset(
    {
        "politics",
        "economy",
        "world",
        "conflict",
        "sports",
        "technology",
        "science",
        "health",
        "culture",
        "society",
        "general",
    }
)

# Source-grounded guardrail (product invariant): the digest must synthesize ONLY
# what the posts say — no added facts, inference, or editorializing. Wahb stays a
# source-grounded aggregator, not an AI news anchor. The category is a
# CLASSIFICATION of the event, not added information.
SYSTEM_PROMPT = """
You are a news desk editor. You are given several short posts from DIFFERENT
sources, all about the SAME news event. Produce a brief digest of the event.

Rules:
- Write in the SAME language as the posts (Arabic posts => Arabic digest).
- "summary": ONE neutral sentence stating what happened.
- "bullets": exactly {n} short, factual points, each one line, most important
  first. Each bullet is a distinct fact about the event.
- Use ONLY information explicitly present in the posts. Do NOT add facts, infer,
  speculate, or editorialize. If the posts disagree, state the event neutrally.
- No source names, no hashtags, no opinions, no first person, no trailing
  punctuation beyond a period.
- "category": classify the event into EXACTLY ONE of these English slugs:
  politics, economy, world, conflict, sports, technology, science, health,
  culture, society, general. Pick the single best fit; use "general" only when
  none clearly applies. The slug is always English even for Arabic posts.

Return ONLY a JSON object:
{{"summary": "...", "bullets": ["...", "..."], "category": "..."}}
""".strip()

MAX_SNIPPETS = 12
MAX_SNIPPET_CHARS = 600
MAX_SUMMARY_CHARS = 280
MAX_BULLET_CHARS = 200


class TopicDigestService:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def digest(self, texts: list[str], max_bullets: int = 3) -> TopicDigestResponse:
        snippets = [t.strip()[:MAX_SNIPPET_CHARS] for t in texts if t and t.strip()][
            :MAX_SNIPPETS
        ]
        joined = "\n\n---\n\n".join(snippets)
        system = SYSTEM_PROMPT.format(n=max_bullets)
        user_prompt = f"Posts:\n\n{joined}"

        raw = await self.llm.complete(
            system,
            user_prompt,
            max_tokens=400,
            temperature=0.2,
            operation="topic_digest",
        )

        summary = ""
        bullets: list[str] = []
        category = "general"
        try:
            parsed = json.loads(_strip_fences(raw))
            summary = str(parsed.get("summary", "")).strip()
            raw_bullets = parsed.get("bullets") or []
            if isinstance(raw_bullets, list):
                bullets = [str(b).strip() for b in raw_bullets if str(b).strip()]
            cat = str(parsed.get("category", "")).strip().lower()
            if cat in CATEGORIES:
                category = cat
        except Exception:
            # Model returned prose instead of JSON — salvage non-empty lines as bullets.
            lines = [ln.strip(" -•\t").strip() for ln in _strip_fences(raw).splitlines()]
            bullets = [ln for ln in lines if ln][:max_bullets]

        # Normalize: strip bullet markers, clamp count + lengths.
        cleaned = []
        for b in bullets:
            b = b.lstrip("-•*0123456789. ").strip()
            if b:
                cleaned.append(b[:MAX_BULLET_CHARS])
        bullets = cleaned[:max_bullets]
        summary = summary[:MAX_SUMMARY_CHARS]

        return TopicDigestResponse(summary=summary, bullets=bullets, category=category)
