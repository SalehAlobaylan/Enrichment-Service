"""Model manager for Enrichment-Service.

This service hosts two text-side models: the Qwen3 dense embedder (retrieval
kNN) and the bge-reranker-v2-m3 cross-encoder (for refining top candidates).
Whisper transcription + CLIP image embedding live in Media-Service.

⚠️  TEMP WORKAROUND (Cranl fixed-instance-RAM limit) — role split:
Cranl won't let us resize one app's RAM, and a single instance can't hold both
fp16 models (~3GB each) without OOM-killing the embedder mid-load. So the
reranker can run as a SEPARATE deployment and the main instance calls it over
HTTP. Behaviour is selected by `ENRICHMENT_ROLE` + `RERANKER_BASE_URL`:

  • role=reranker                      → load ONLY the reranker (serves /v1/rerank)
  • role=api  + RERANKER_BASE_URL set  → load ONLY the embedder; reranker is a
                                         remote HTTP client (RerankerClient)
  • role=api  + RERANKER_BASE_URL ""   → MONOLITH: load both locally (the
                                         original single-instance path; used by
                                         local ./start.sh and any 8GB+ instance)

Collapse the split (delete the reranker app, clear RERANKER_BASE_URL) and the
monolith path takes over with no other changes.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.common.config import Settings
from src.common.utils.logging import get_logger
from src.retrieval.clients.reranker_client import RerankerClient
from src.retrieval.models.embedder import EmbedderWrapper
from src.retrieval.models.reranker import RerankerWrapper

logger = get_logger(__name__)


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._executor = ThreadPoolExecutor(max_workers=2)

        self.role = (settings.ENRICHMENT_ROLE or "api").strip().lower()
        # The "api" role talks to a remote reranker when one is configured;
        # everything else (reranker role, or monolith) uses the in-process
        # wrapper. `reranker` is duck-typed: both expose
        # model_name / is_loaded / load() / rerank(query, candidates).
        self._use_remote_reranker = (
            self.role == "api" and bool(settings.RERANKER_BASE_URL.strip())
        )

        self.embedder = EmbedderWrapper(
            model_name=settings.EMBEDDING_MODEL,
            cache_folder=settings.MODELS_DIR,
            revision=settings.EMBEDDING_MODEL_REVISION,
        )

        if self._use_remote_reranker:
            self.reranker = RerankerClient(
                base_url=settings.RERANKER_BASE_URL.strip(),
                token=settings.SERVICE_AUTH_TOKEN,
                model_name=settings.RERANKER_MODEL,
            )
        else:
            self.reranker = RerankerWrapper(
                model_name=settings.RERANKER_MODEL,
                cache_folder=settings.MODELS_DIR,
            )

    @property
    def is_ready(self) -> dict[str, bool]:
        return {
            "embedder": self.embedder.is_loaded,
            "reranker": self.reranker.is_loaded,
        }

    @property
    def all_ready(self) -> bool:
        # The reranker-role instance has no embedder — its readiness is the
        # reranker. Every other role gates on the embedder; the reranker is
        # best-effort (/v1/related + /v1/feed/news/slide degrade to RRF order
        # when it's unavailable), so it never gates /ready on the api role.
        if self.role == "reranker":
            return self.reranker.is_loaded
        return self.embedder.is_loaded

    async def warmup(self) -> None:
        loop = asyncio.get_event_loop()
        logger.info(
            "loading_models",
            role=self.role,
            remote_reranker=self._use_remote_reranker,
            rerank_enabled=self.settings.RERANK_ENABLED,
        )

        # ── Reranker-only role: load just the cross-encoder, nothing else. ──
        if self.role == "reranker":
            try:
                await loop.run_in_executor(self._executor, self.reranker.load)
                logger.info("model_loaded", model="reranker")
            except Exception as result:  # noqa: BLE001
                logger.error("model_load_failed", model="reranker", error=str(result))
            return

        # ── api role: embedder FIRST and ALONE. ────────────────────────────
        # Loading the embedder on its own (not racing the reranker for RAM)
        # is what prevents the OOM; it also flips /ready green before the
        # slower reranker, so Cranl's health probe passes.
        try:
            await loop.run_in_executor(self._executor, self.embedder.load)
            logger.info("model_loaded", model="embedder")
        except Exception as result:  # noqa: BLE001
            logger.error("model_load_failed", model="embedder", error=str(result))

        # Remote reranker → nothing to load here (it's a different deployment).
        if self._use_remote_reranker:
            self.reranker.load()  # logs "reranker_remote_configured"
            return

        # Monolith fallback: load the local reranker too, unless disabled.
        if not self.settings.RERANK_ENABLED:
            logger.info("reranker_disabled_skipping_load")
            return
        try:
            await loop.run_in_executor(self._executor, self.reranker.load)
            logger.info("model_loaded", model="reranker")
        except Exception as result:  # noqa: BLE001
            logger.error("model_load_failed", model="reranker", error=str(result))
