import pytest
from pydantic import ValidationError

from src.extraction.schemas.extract import (
    ExtractRequest,
    TwitterRecommendationsRequest,
    YouTubeResolveLinksRequest,
)
from src.llm.schemas.classify import AccountClassifyRequest, AccountToClassify
from src.llm.schemas.summarize import SummarizeRequest
from src.llm.schemas.topic_digest import TopicDigestRequest
from src.llm.schemas.translate import TranslateRequest
from src.retrieval.schemas.embed import EmbedRequest
from src.retrieval.schemas.related import RelatedRequest
from src.retrieval.schemas.rerank import RerankRequest


def test_embed_rejects_mismatched_or_duplicate_writeback_ids() -> None:
    with pytest.raises(ValidationError, match="match texts length"):
        EmbedRequest(texts=["one", "two"], content_ids=["one"])
    with pytest.raises(ValidationError, match="unique"):
        EmbedRequest(texts=["one", "two"], content_ids=["same", "same"])


def test_embed_rejects_overlong_or_oversized_batches() -> None:
    with pytest.raises(ValidationError, match="12000"):
        EmbedRequest(texts=["x" * 12_001])
    with pytest.raises(ValidationError):
        EmbedRequest(texts=["x"] * 33)


def test_rerank_rejects_empty_and_oversized_candidates() -> None:
    with pytest.raises(ValidationError):
        RerankRequest(query="query", candidates=[])
    with pytest.raises(ValidationError):
        RerankRequest(query="query", candidates=["x"] * 33)
    with pytest.raises(ValidationError, match="4000"):
        RerankRequest(query="query", candidates=["x" * 4_001])


def test_related_rejects_ambiguous_or_unbounded_filters() -> None:
    with pytest.raises(ValidationError, match="content_id OR text"):
        RelatedRequest(content_id="one", text="two")
    with pytest.raises(ValidationError, match="unique"):
        RelatedRequest(text="query", exclude_ids=["same", "same"])
    with pytest.raises(ValidationError):
        RelatedRequest(text="query", exclude_ids=["x"] * 201)
    with pytest.raises(ValidationError):
        RelatedRequest(text="query", types=["UNKNOWN"])
    with pytest.raises(ValidationError):
        RelatedRequest(text="query", types=["TWEET"])
    with pytest.raises(ValidationError):
        RelatedRequest(text="query", formats=["VIDEO"])


def test_llm_schemas_reject_oversized_workloads() -> None:
    with pytest.raises(ValidationError):
        AccountClassifyRequest(
            accounts=[AccountToClassify(handle=f"account-{index}") for index in range(51)]
        )
    with pytest.raises(ValidationError):
        SummarizeRequest(text="x" * 12_001)
    with pytest.raises(ValidationError):
        TranslateRequest(text="x" * 12_001)
    with pytest.raises(ValidationError, match="6000"):
        TopicDigestRequest(texts=["x" * 6_001])


def test_extraction_schemas_reject_unbounded_inputs() -> None:
    with pytest.raises(ValidationError):
        ExtractRequest(url="x" * 2_049)
    with pytest.raises(ValidationError):
        TwitterRecommendationsRequest(seed="seed", limit=41)
    with pytest.raises(ValidationError):
        YouTubeResolveLinksRequest(inputs=["https://youtube.test"] * 51)
