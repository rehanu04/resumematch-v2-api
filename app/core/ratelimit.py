import time
from collections import defaultdict, deque
from fastapi import Request, HTTPException

# Simple in-memory sliding window rate limiter
WINDOW_SECONDS = 60

# tune these:
MAX_PER_IP = 60
MAX_PER_KEY = 120

_ip_hits = defaultdict(deque)
_key_hits = defaultdict(deque)

def _prune(q: deque, now: float):
    cutoff = now - WINDOW_SECONDS
    while q and q[0] < cutoff:
        q.popleft()

async def rate_limit(request: Request):
    now = time.time()

    ip = request.client.host if request.client else "unknown"
    key = request.headers.get("X-App-Key", "no-key")

    iq = _ip_hits[ip]
    kq = _key_hits[key]

    _prune(iq, now)
    _prune(kq, now)

    if len(iq) >= MAX_PER_IP:
        raise HTTPException(status_code=429, detail="Rate limit exceeded (IP)")
    if len(kq) >= MAX_PER_KEY:
        raise HTTPException(status_code=429, detail="Rate limit exceeded (Key)")

    iq.append(now)
    kq.append(now)


def debug_state():
    return {"ip_keys": len(_ip_hits), "key_keys": len(_key_hits)}

