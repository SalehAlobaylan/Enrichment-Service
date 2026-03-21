from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.whisper import TranscribeResult, WhisperWrapper
from src.services.transcription import TranscriptionService


@pytest.fixture
def mock_whisper() -> MagicMock:
    whisper = MagicMock(spec=WhisperWrapper)
    whisper.model_size = "base"
    whisper.transcribe.return_value = TranscribeResult(
        text="Hello world",
        language="en",
        language_probability=0.98,
        segments=[{"start": 0.0, "end": 1.5, "text": "Hello world"}],
        duration_sec=1.5,
    )
    return whisper


@pytest.fixture
def mock_cms() -> AsyncMock:
    cms = AsyncMock()
    cms.create_transcript.return_value = {"id": "tr-123"}
    cms.link_transcript.return_value = {"ok": True}
    return cms


@pytest.fixture
def service(mock_whisper: MagicMock, mock_cms: AsyncMock) -> TranscriptionService:
    return TranscriptionService(mock_whisper, mock_cms)


@pytest.mark.asyncio
async def test_transcribe_returns_response(service: TranscriptionService) -> None:
    result = await service.transcribe_file("/tmp/test.mp3")
    assert result.text == "Hello world"
    assert result.language == "en"
    assert result.duration_sec == 1.5
    assert len(result.segments) == 1


@pytest.mark.asyncio
async def test_no_writeback_without_content_id(
    service: TranscriptionService, mock_cms: AsyncMock
) -> None:
    await service.transcribe_file("/tmp/test.mp3")
    mock_cms.create_transcript.assert_not_called()


@pytest.mark.asyncio
async def test_writeback_with_content_id(
    service: TranscriptionService, mock_cms: AsyncMock
) -> None:
    await service.transcribe_file("/tmp/test.mp3", content_id="content-456")
    mock_cms.create_transcript.assert_called_once()
    mock_cms.link_transcript.assert_called_once_with("content-456", "tr-123")


@pytest.mark.asyncio
async def test_writeback_failure_does_not_raise(
    service: TranscriptionService, mock_cms: AsyncMock
) -> None:
    mock_cms.create_transcript.side_effect = Exception("CMS down")
    # Should not raise — write-back failures are logged, not propagated
    result = await service.transcribe_file("/tmp/test.mp3", content_id="content-789")
    assert result.text == "Hello world"
