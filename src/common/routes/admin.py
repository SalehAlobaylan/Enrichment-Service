import asyncio
import os
import signal

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from src.common.auth.service_auth import verify_restart_token
from src.common.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Module seam for deterministic route tests. Production retains os.kill.
terminate_process = os.kill


async def _shutdown_after_delay() -> None:
    await asyncio.sleep(0.25)
    logger.info("restart_requested", source="/v1/admin/restart")
    terminate_process(os.getpid(), signal.SIGTERM)


@router.post("/admin/restart", status_code=202, dependencies=[Depends(verify_restart_token)])
async def restart(background_tasks: BackgroundTasks, request: Request) -> dict[str, str]:
    """Self-restart: sends SIGTERM so FastAPI's lifespan shutdown runs cleanly.

    The process must be supervised (Cranl, k8s, systemd, Docker
    restart:always) for it to actually come back up.
    """
    operator = request.headers.get("X-Operator-Email", "unknown")[:256]
    request_id = request.headers.get("X-Request-ID", "unknown")[:128]
    logger.info("restart_accepted", operator=operator, request_id=request_id)
    background_tasks.add_task(_shutdown_after_delay)
    return {
        "message": "Restart accepted. Service is shutting down — supervisor must bring it back.",
        "service": "enrichment",
    }
