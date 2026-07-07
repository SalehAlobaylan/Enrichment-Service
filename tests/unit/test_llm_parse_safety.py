from unittest.mock import AsyncMock

import pytest

from src.common.middleware.error_handler import LLMError
from src.llm.schemas.chapters import ChaptersGenerateRequest, ChapterWindow
from src.llm.services.chapters import ChaptersGenerationService
from src.llm.services.summarization import SummarizationService
from src.llm.services.translation import TranslationService


@pytest.mark.asyncio
async def test_summarization_accepts_valid_json() -> None:
    llm = AsyncMock()
    cms = AsyncMock()
    llm.complete.return_value = '{"summary": "Short version.", "key_points": ["a"]}'
    service = SummarizationService(llm, cms)

    result = await service.summarize("Long text")

    assert result.summary == "Short version."
    assert result.key_points == ["a"]


@pytest.mark.asyncio
async def test_summarization_rejects_garbage_output() -> None:
    llm = AsyncMock()
    cms = AsyncMock()
    llm.complete.return_value = "Sure, here is a summary without JSON."
    service = SummarizationService(llm, cms)

    with pytest.raises(LLMError, match="malformed JSON"):
        await service.summarize("Long text")


@pytest.mark.asyncio
async def test_translation_accepts_valid_json() -> None:
    llm = AsyncMock()
    cms = AsyncMock()
    llm.complete.return_value = '{"translated_text": "Hello", "source_language": "ar"}'
    service = TranslationService(llm, cms)

    result = await service.translate("hola", target_language="en")

    assert result.translated_text == "Hello"
    assert result.source_language == "ar"
    assert result.target_language == "en"


@pytest.mark.asyncio
async def test_translation_rejects_garbage_output() -> None:
    llm = AsyncMock()
    cms = AsyncMock()
    llm.complete.return_value = "Translation: Hello"
    service = TranslationService(llm, cms)

    with pytest.raises(LLMError, match="malformed JSON"):
        await service.translate("hola", target_language="en")


@pytest.mark.asyncio
async def test_chapters_rejects_garbage_output() -> None:
    llm = AsyncMock()
    llm.complete.return_value = "chapter one starts at the beginning"
    service = ChaptersGenerationService(llm)

    with pytest.raises(LLMError, match="malformed JSON"):
        await service.generate(_chapters_request())


@pytest.mark.asyncio
async def test_chapters_accepts_valid_empty_chapters() -> None:
    llm = AsyncMock()
    llm.complete.return_value = '{"chapters": []}'
    service = ChaptersGenerationService(llm)

    result = await service.generate(_chapters_request())

    assert result.chapters == []


def _chapters_request() -> ChaptersGenerateRequest:
    return ChaptersGenerateRequest(
        windows=[
            ChapterWindow(index=0, start_sec=0, text="Intro"),
            ChapterWindow(index=1, start_sec=300, text="Main topic"),
        ]
    )
