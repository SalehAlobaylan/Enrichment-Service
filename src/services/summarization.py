import json

from src.clients.cms import CMSClient
from src.clients.llm import LLMClient
from src.schemas.summarize import SummarizeResponse
from src.utils.logging import get_logger
from src.utils.metrics import summarizations_total

logger = get_logger(__name__)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    return text.strip()


SYSTEM_PROMPT = """You are a content summarizer. Summarize the given text.
Return ONLY a JSON object with these fields:
- "summary": a concise summary
- "key_points": an array of 3-5 key points as strings

Do not include any other text or explanation."""


class SummarizationService:
    def __init__(self, llm_client: LLMClient, cms_client: CMSClient):
        self.llm = llm_client
        self.cms_client = cms_client

    async def summarize(
        self,
        text: str,
        max_length: int = 200,
        style: str = "brief",
        content_id: str | None = None,
    ) -> SummarizeResponse:
        user_prompt = (
            f"Summarize the following text in approximately {max_length} words. "
            f"Style: {style}.\n\n{text}"
        )

        raw = await self.llm.complete(
            SYSTEM_PROMPT, user_prompt, max_tokens=1024, operation="summarize"
        )

        parsed = json.loads(_strip_fences(raw))

        summarizations_total.labels(status="success").inc()

        response = SummarizeResponse(
            summary=parsed["summary"],
            key_points=parsed.get("key_points", []),
        )

        if content_id:
            await self._write_back(content_id, response)

        return response

    async def _write_back(self, content_id: str, result: SummarizeResponse) -> None:
        try:
            await self.cms_client.update_content(
                content_id,
                metadata={
                    "summary": result.summary,
                    "key_points": result.key_points,
                },
            )
            logger.info("summary_writeback_complete", content_id=content_id)
        except Exception as exc:
            logger.error(
                "summary_writeback_failed",
                content_id=content_id,
                error=str(exc),
            )
