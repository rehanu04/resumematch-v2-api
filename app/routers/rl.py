from fastapi import APIRouter, Depends, Request
from app.core.security import require_app_key
from app.core import ratelimit

router = APIRouter(prefix="/v1/rl", tags=["ratelimit"])

@router.get("/state")
def state(request: Request, _=Depends(require_app_key)):
    ip = request.client.host if request.client else "unknown"
    key = request.headers.get("X-App-Key", "no-key")

    # Get current queue sizes for this IP/key
    ipq = ratelimit._ip_hits.get(ip)
    kq = ratelimit._key_hits.get(key)

    return {
        "client_ip": ip,
        "client_key_prefix": key[:6],
        "window_seconds": ratelimit.WINDOW_SECONDS,
        "max_per_ip": ratelimit.MAX_PER_IP,
        "max_per_key": ratelimit.MAX_PER_KEY,
        "current_ip_hits": len(ipq) if ipq is not None else 0,
        "current_key_hits": len(kq) if kq is not None else 0,
        "seen": ratelimit.debug_state(),
    }
