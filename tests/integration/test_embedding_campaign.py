import json
from unittest.mock import MagicMock

import pytest
import respx
from httpx import Response

from src.common.clients.cms import CMSClient
from src.common.config import Settings

CONTENT_ID = "dba3dd23-9718-4a69-a06d-7551f22ef271"
REQUEST_ID = "dba3dd23-9718-4a69-a06d-7551f22ef272"
CAMPAIGN = {"campaign_id": 17, "action_id": 23, "request_id": REQUEST_ID}


@pytest.mark.parametrize("campaign", [CAMPAIGN, None])
def test_embedding_writeback_preserves_campaign_without_changing_legacy_calls(
    client,
    auth_headers,
    mock_cms_client,
    campaign,
):
    client.app.state.model_manager.embedder.encode = MagicMock(
        return_value={"dense": [[0.1] * 1024], "sparse": None}
    )
    payload = {"texts": ["source text"], "content_ids": [CONTENT_ID]}
    if campaign is not None:
        payload["embedding_campaign"] = campaign

    response = client.post("/v1/embed", json=payload, headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["write_back_status"] == "ok"
    mock_cms_client.store_embedding.assert_awaited_once()
    kwargs = mock_cms_client.store_embedding.call_args.kwargs
    if campaign is None:
        assert "embedding_campaign" not in kwargs
    else:
        assert kwargs["embedding_campaign"] == campaign
    assert kwargs["content_stage"] is None
    assert kwargs["artifact_recovery"] is None
    assert kwargs["pipeline_repair"] is None


@pytest.mark.parametrize("correlation", ["content_stage", "artifact_recovery", "pipeline_repair"])
def test_embedding_campaign_rejects_other_write_correlations_before_inference(
    client,
    auth_headers,
    correlation,
):
    other = {
        "request_id": REQUEST_ID,
        "attempt_id": REQUEST_ID,
        "claim_token": REQUEST_ID,
        "fence_token": REQUEST_ID,
        "input_digest": "a" * 64,
        "input_fingerprint": "a" * 64,
        "producer_event_id": REQUEST_ID,
        "repair_id": REQUEST_ID,
        "expected_item_version": "2026-11-01T00:00:00Z",
    }
    payload = {
        "texts": ["source text"],
        "content_ids": [CONTENT_ID],
        "embedding_campaign": CAMPAIGN,
        correlation: other,
    }
    if correlation == "content_stage":
        payload["lane"] = "news"

    response = client.post("/v1/embed", json=payload, headers=auth_headers)

    assert response.status_code == 422
    client.app.state.model_manager.embedder.encode.assert_not_called()


@pytest.mark.parametrize("content_ids", [None, [CONTENT_ID, REQUEST_ID]])
def test_embedding_campaign_requires_exactly_one_target(client, auth_headers, content_ids):
    payload = {"texts": ["source text"], "embedding_campaign": CAMPAIGN}
    if content_ids is not None:
        payload["content_ids"] = content_ids
        payload["texts"] *= len(content_ids)
    response = client.post("/v1/embed", json=payload, headers=auth_headers)
    assert response.status_code == 422
    client.app.state.model_manager.embedder.encode.assert_not_called()


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("campaign", [CAMPAIGN, None])
async def test_cms_embedding_client_preserves_opaque_campaign(campaign):
    cms = CMSClient(Settings(CMS_BASE_URL="http://cms.test", CMS_SERVICE_TOKEN="test-token"))
    route = respx.patch(f"http://cms.test/internal/content-items/{CONTENT_ID}/embedding").mock(
        return_value=Response(200, json={"ok": True})
    )
    try:
        await cms.store_embedding(CONTENT_ID, [0.1], embedding_campaign=campaign)
        payload = json.loads(route.calls[0].request.content)
        if campaign is None:
            assert "embedding_campaign" not in payload
        else:
            assert payload["embedding_campaign"] == campaign
    finally:
        await cms.close()
