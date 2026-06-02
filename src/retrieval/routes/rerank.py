"""POST /v1/rerank — internal cross-encoder scoring endpoint.

╔══════════════════════════════════════════════════════════════════════════╗
║ ⚠️  TEMP WORKAROUND — DELETE WHEN THE INSTANCE-SIZE LIMIT IS GONE          ║
╠══════════════════════════════════════════════════════════════════════════╣
║ This endpoint ONLY exists because Cranl does not allow resizing a single  ║
║ app's RAM, and one instance cannot hold BGE-M3 + bge-reranker-v2-m3 at    ║
║ once (loading both OOM-kills the embedder). So the reranker runs as its    ║
║ OWN Cranl deployment (ENRICHMENT_ROLE=reranker) that exposes this         ║
║ endpoint, and the main "api" instance calls it over HTTP for the rerank   ║
║ stage of /v1/related and /v1/feed/news/slide.                             ║
║                                                                            ║
║ It is INTERNAL/service-to-service only (same SERVICE_AUTH_TOKEN). It is    ║
║ NOT mounted on the "api" role — only on the reranker-role instance.       ║
║                                                                            ║
║ When a single instance can hold both models (~8GB) again: clear           ║
║ RERANKER_BASE_URL, drop the reranker deployment, and this route + its     ║
║ client (clients/reranker_client.py) can be removed entirely. The          ║
║ in-process RerankerWrapper path is unchanged underneath.                  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
from fastapi import APIRouter, Depends, Request

from src.common.auth.service_auth import verify_service_token
from src.common.middleware.error_handler import EmbeddingError
from src.common.utils.logging import get_logger
from src.retrieval.schemas.rerank import RerankRequest, RerankResponse

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/rerank", response_model=RerankResponse)
async def rerank(body: RerankRequest, request: Request) -> RerankResponse:
    reranker = request.app.state.model_manager.reranker

    if not reranker.is_loaded:
        # On the reranker-role instance this means the model is still warming;
        # the caller (RelatedService) treats any rerank failure as "skip rerank,
        # keep RRF order", so a 503-ish signal here is safe.
        raise EmbeddingError("Reranker model is not loaded")

    if not body.candidates:
        return RerankResponse(scores=[], model=reranker.model_name)

    import asyncio

    scores = await asyncio.to_thread(reranker.rerank, body.query, body.candidates)
    return RerankResponse(scores=[float(s) for s in scores], model=reranker.model_name)
