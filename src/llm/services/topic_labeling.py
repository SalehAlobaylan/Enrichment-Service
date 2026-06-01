import json

from src.common.utils.logging import get_logger
from src.llm.clients.llm import LLMClient
from src.llm.schemas.topic_label import TopicLabelResponse

logger = get_logger(__name__)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    return text.strip()


SYSTEM_PROMPT = """
You are a news desk editor. You are given one or more short news article
snippets that belong to the same story or theme. Write ONE concise, meaningful
topic title that captures the SPECIFIC theme they share.

Rules:
- A clear noun phrase of 3 to 8 words — NOT a single keyword, NOT a vague word
  like "news" or "money".
- Write it in the SAME language as the snippets (Arabic snippets => Arabic
  title; English => English).
- Be specific (e.g. "New debt enforcement and salary garnishment rules",
  not "enforcement"; "نظام التنفيذ وحجز الرواتب", not "تنفيذ").
- No hashtags, no surrounding quotes, no trailing punctuation.

Return ONLY a JSON object: {"label": "..."}
""".strip()

MAX_SNIPPETS = 8
MAX_SNIPPET_CHARS = 500
MAX_LABEL_CHARS = 120


class TopicLabelingService:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def label(self, texts: list[str]) -> TopicLabelResponse:
        snippets = [
            t.strip()[:MAX_SNIPPET_CHARS] for t in texts if t and t.strip()
        ][:MAX_SNIPPETS]
        joined = "\n\n---\n\n".join(snippets)
        user_prompt = f"Article snippets:\n\n{joined}"

        raw = await self.llm.complete(
            SYSTEM_PROMPT,
            user_prompt,
            max_tokens=60,
            temperature=0.2,
            operation="topic_label",
        )

        label = ""
        try:
            parsed = json.loads(_strip_fences(raw))
            label = str(parsed.get("label", "")).strip()
        except Exception:
            # Model returned a bare string instead of JSON — salvage it.
            label = _strip_fences(raw).strip().strip('"').strip()

        # Last-resort fallback so we never create an empty topic label.
        if not label and snippets:
            label = snippets[0][:60].strip()

        return TopicLabelResponse(label=label[:MAX_LABEL_CHARS])
