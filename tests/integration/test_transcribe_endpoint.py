import io
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.models.whisper import TranscribeResult


def test_transcribe_file_upload(client: TestClient, auth_headers: dict[str, str]) -> None:
    client.app.state.model_manager.whisper.transcribe = MagicMock(  # type: ignore[union-attr]
        return_value=TranscribeResult(
            text="Transcribed text",
            language="en",
            language_probability=0.95,
            segments=[{"start": 0.0, "end": 2.0, "text": "Transcribed text"}],
            duration_sec=2.0,
        )
    )
    audio_data = io.BytesIO(b"\x00" * 100)
    resp = client.post(
        "/v1/transcribe",
        files={"audio_file": ("test.mp3", audio_data, "audio/mpeg")},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Transcribed text"
    assert data["language"] == "en"
    assert len(data["segments"]) == 1


def test_transcribe_requires_file_or_url(client: TestClient, auth_headers: dict[str, str]) -> None:
    resp = client.post("/v1/transcribe", headers=auth_headers)
    assert resp.status_code == 422


def test_transcribe_requires_auth(client: TestClient) -> None:
    resp = client.post("/v1/transcribe")
    assert resp.status_code in (401, 403)
