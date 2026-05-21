from fastapi import APIRouter, Depends, Request

from src.auth.service_auth import verify_service_token
from src.middleware.error_handler import EmbeddingError
from src.schemas.embed import EmbedQueryRequest, EmbedQueryResponse, EmbedRequest, EmbedResponse
from src.services.embedding import EmbeddingService
from src.services.tagging import TaggingService
from src.utils.logging import get_logger
from src.utils.metrics import embeddings_total

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/embed", response_model=EmbedResponse)
async def embed(body: EmbedRequest, request: Request) -> EmbedResponse:
    model_manager = request.app.state.model_manager
    cms_client = request.app.state.cms_client
    llm_client = request.app.state.llm_client
    tagger = TaggingService(llm_client) if body.extract_tags else None
    service = EmbeddingService(model_manager.embedder, cms_client, tagger=tagger)

    if not model_manager.embedder.is_loaded:
        raise EmbeddingError("Embedding model is not loaded")

    if body.content_ids and len(body.content_ids) != len(body.texts):
        raise EmbeddingError("content_ids length must match texts length")

    try:
        return await service.embed(
            body.texts,
            content_ids=body.content_ids,
            extract_tags=body.extract_tags,
        )
    except EmbeddingError:
        raise
    except Exception as exc:
        embeddings_total.labels(status="failure").inc()
        logger.error("embedding_failed", error=str(exc))
        raise EmbeddingError(f"Embedding failed: {exc}") from exc


@router.post("/embed/query", response_model=EmbedQueryResponse)
async def embed_query(body: EmbedQueryRequest, request: Request) -> EmbedQueryResponse:
    model_manager = request.app.state.model_manager
    cms_client = request.app.state.cms_client
    service = EmbeddingService(model_manager.embedder, cms_client)

    if not model_manager.embedder.is_loaded:
        raise EmbeddingError("Embedding model is not loaded")

    try:
        return await service.embed_query(body.text)
    except EmbeddingError:
        raise
    except Exception as exc:
        embeddings_total.labels(status="failure").inc()
        logger.error("embed_query_failed", error=str(exc))
        raise EmbeddingError(f"Embedding failed: {exc}") from exc
