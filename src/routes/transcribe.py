import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from src.auth.service_auth import verify_service_token
from src.middleware.error_handler import TranscriptionError
from src.schemas.transcribe import TranscribeResponse
from src.services.transcription import TranscriptionService
from src.utils.logging import get_logger
from src.utils.metrics import transcriptions_total

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])

TEMP_DIR = os.environ.get("MEDIA_TEMP_DIR", tempfile.gettempdir())
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB


async def _spool_upload_to_disk(
    upload: UploadFile, tmp_path: str, max_bytes: int
) -> int:
    """Stream upload to disk; abort if it exceeds max_bytes. Returns bytes written."""
    written = 0
    with open(tmp_path, "wb") as f:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise TranscriptionError(
                    f"Upload exceeds maximum size of {max_bytes // (1024 * 1024)} MB"
                )
            f.write(chunk)
    return written


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: Request,
    audio_file: UploadFile | None = File(None),
    url: str | None = Form(None),
    content_id: str | None = Form(None),
    language: str | None = Form(None),
    word_timestamps: bool = Form(False),
) -> TranscribeResponse:
    settings = request.app.state.settings
    model_manager = request.app.state.model_manager
    cms_client = request.app.state.cms_client
    service = TranscriptionService(model_manager.whisper, cms_client)

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    if not model_manager.whisper.is_loaded:
        raise TranscriptionError("Whisper model is not loaded")

    # Fast path: reject oversize uploads via Content-Length before streaming.
    if audio_file is not None:
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > max_bytes:
            raise TranscriptionError(
                f"Upload exceeds maximum size of {settings.MAX_UPLOAD_MB} MB"
            )

    try:
        if audio_file and audio_file.filename:
            suffix = os.path.splitext(audio_file.filename)[1] or ".mp3"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=TEMP_DIR)
            os.close(fd)  # we'll reopen via _spool_upload_to_disk
            try:
                await _spool_upload_to_disk(audio_file, tmp_path, max_bytes)

                return await service.transcribe_file(
                    tmp_path,
                    content_id=content_id,
                    language=language,
                    word_timestamps=word_timestamps,
                )
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        elif url:
            return await service.transcribe_url(
                url,
                content_id=content_id,
                language=language,
                word_timestamps=word_timestamps,
            )

        else:
            raise TranscriptionError("Provide either 'audio_file' or 'url'")

    except TranscriptionError:
        raise
    except Exception as exc:
        transcriptions_total.labels(
            status="failure", model_size=model_manager.whisper.model_size
        ).inc()
        logger.error("transcription_failed", error=str(exc))
        raise TranscriptionError(f"Transcription failed: {exc}") from exc
