import hmac

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()


async def verify_service_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    token = request.app.state.settings.service_auth_token
    if not token:
        raise HTTPException(status_code=500, detail="SERVICE_AUTH_TOKEN not configured")
    if credentials.credentials != token:
        raise HTTPException(status_code=401, detail="Invalid service token")
    return credentials.credentials


async def verify_restart_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Authorize only the separate, human-gated restart capability."""
    token = request.app.state.settings.ENRICHMENT_RESTART_TOKEN
    if not token:
        raise HTTPException(status_code=503, detail="Restart capability is not configured")
    if not hmac.compare_digest(credentials.credentials, token):
        raise HTTPException(status_code=401, detail="Invalid restart capability")
    return credentials.credentials
