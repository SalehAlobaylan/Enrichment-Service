import asyncio
import os
import signal

from fastapi import APIRouter, BackgroundTasks, Depends

from src.common.auth.service_auth import verify_service_token
from src.common.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(verify_service_token)])


async def _shutdown_after_delay() -> None:
    await asyncio.sleep(0.25)
    logger.info("restart_requested", source="/v1/admin/restart")
    os.kill(os.getpid(), signal.SIGTERM)


@router.post("/admin/restart", status_code=202)
async def restart(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Self-restart: sends SIGTERM so FastAPI's lifespan shutdown runs cleanly.

    The process must be supervised (Cranl, k8s, systemd, Docker
    restart:always) for it to actually come back up.
    """
    background_tasks.add_task(_shutdown_after_delay)
    return {
        "message": "Restart accepted. Service is shutting down — supervisor must bring it back.",
        "service": "enrichment",
    }
