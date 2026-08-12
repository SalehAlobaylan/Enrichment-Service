import json
from json import JSONDecodeError

from src.common.clients.cms import CMSClient
from src.common.middleware.error_handler import LLMError
from src.common.utils.logging import get_logger
from src.common.utils.metrics import summarizations_total
from src.llm.clients.llm import LLMClient
from src.llm.schemas.summarize import SummarizeResponse

logger = get_logger(__name__)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1 :] if "\n" in text else text[3:]
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
        artifact_recovery: dict[str, str] | None = None,
    ) -> SummarizeResponse:
        user_prompt = (
            f"Summarize the following text in approximately {max_length} words. "
            f"Style: {style}.\n\n{text}"
        )

        raw = await self.llm.complete(
            SYSTEM_PROMPT, user_prompt, max_tokens=1024, operation="summarize"
        )

        parsed = self._parse(raw)

        summarizations_total.labels(status="success").inc()

        response = SummarizeResponse(
            summary=parsed["summary"],
            key_points=parsed.get("key_points", []),
        )

        if content_id:
            await self._write_back(content_id, response, artifact_recovery)

        return response

    def _parse(self, raw: str) -> dict[str, object]:
        try:
            parsed = json.loads(_strip_fences(raw))
        except JSONDecodeError as exc:
            summarizations_total.labels(status="failure").inc()
            logger.warning("summary_parse_failed", raw_preview=raw[:200])
            raise LLMError("Summarization returned malformed JSON") from exc

        if (
            not isinstance(parsed, dict)
            or not isinstance(parsed.get("summary"), str)
            or not parsed["summary"].strip()
        ):
            summarizations_total.labels(status="failure").inc()
            logger.warning("summary_parse_missing_required_fields")
            raise LLMError("Summarization returned an invalid response")

        return parsed

    async def _write_back(
        self,
        content_id: str,
        result: SummarizeResponse,
        artifact_recovery: dict[str, str] | None = None,
    ) -> None:
        try:
            await self.cms_client.merge_enrichment_metadata(
                content_id,
                {
                    "summary": result.summary,
                    "key_points": result.key_points,
                },
                artifact_recovery=artifact_recovery,
            )
            result.write_back_status = "persisted"
            logger.info("summary_writeback_complete", content_id=content_id)
        except Exception:
            result.write_back_status = "failed"
            result.write_back_error = "cms_writeback_failed"
            logger.error(
                "summary_writeback_failed",
                content_id=content_id,
            )
