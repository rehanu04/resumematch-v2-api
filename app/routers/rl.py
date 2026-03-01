from fastapi import APIRouter, Depends, Request
from app.core.security import require_app_key
from app.core import ratelimit

router = APIRouter(prefix="/v1/rl", tags=["ratelimit"])

@router.get("/state")
def state(request: Request, _=Depends(require_app_key)):
    ip = request.client.host if request.client else "unknown"
    key = request.headers.get("X-App-Key", "no-key")
    return {
        "client_ip": ip,
        "client_key_prefix": key[:6],
        "window_seconds": ratelimit.WINDOW_SECONDS,
        "max_per_ip": ratelimit.MAX_PER_IP,
        "max_per_key": ratelimit.MAX_PER_KEY,
        "seen": ratelimit.debug_state(),
    }
