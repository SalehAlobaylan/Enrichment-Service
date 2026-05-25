import json

from src.common.clients.cms import CMSClient
from src.common.utils.logging import get_logger
from src.common.utils.metrics import translations_total
from src.llm.clients.llm import LLMClient
from src.llm.schemas.translate import TranslateResponse

logger = get_logger(__name__)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    return text.strip()


SYSTEM_PROMPT = """
You are a professional translator. Translate the given text to the target language.
Return ONLY a JSON object with these fields:
- "translated_text": the translation
- "source_language": ISO 639-1 code of the detected source language

Do not include any other text or explanation.
""".strip()


class TranslationService:
    def __init__(self, llm_client: LLMClient, cms_client: CMSClient):
        self.llm = llm_client
        self.cms_client = cms_client

    async def translate(
        self,
        text: str,
        target_language: str = "en",
        source_language: str | None = None,
        content_id: str | None = None,
    ) -> TranslateResponse:
        source_hint = f" (source language: {source_language})" if source_language else ""
        user_prompt = f"Translate the following text to {target_language}{source_hint}:\n\n{text}"

        raw = await self.llm.complete(
            SYSTEM_PROMPT, user_prompt, max_tokens=2048, operation="translate"
        )

        parsed = json.loads(_strip_fences(raw))
        detected_source = parsed.get("source_language", source_language or "unknown")

        translations_total.labels(status="success").inc()

        response = TranslateResponse(
            translated_text=parsed["translated_text"],
            source_language=detected_source,
            target_language=target_language,
        )

        if content_id:
            await self._write_back(content_id, response)

        return response

    async def _write_back(self, content_id: str, result: TranslateResponse) -> None:
        try:
            key = f"translation_{result.target_language}"
            await self.cms_client.update_content(
                content_id,
                metadata={key: result.translated_text},
            )
            logger.info("translation_writeback_complete", content_id=content_id)
        except Exception as exc:
            logger.error(
                "translation_writeback_failed",
                content_id=content_id,
                error=str(exc),
            )
