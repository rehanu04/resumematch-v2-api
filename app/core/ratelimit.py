import time
from collections import defaultdict, deque
from fastapi import Request, HTTPException

WINDOW_SECONDS = 60

# Key-based only (reliable behind proxies)
MAX_PER_KEY = 30

_key_hits = defaultdict(deque)

def _prune(q: deque, now: float):
    cutoff = now - WINDOW_SECONDS
    while q and q[0] < cutoff:
        q.popleft()

async def rate_limit(request: Request):
    now = time.time()
    key = request.headers.get("X-App-Key", "no-key")

    q = _key_hits[key]
    _prune(q, now)

    if len(q) >= MAX_PER_KEY:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    q.append(now)

def debug_state():
    return {"key_keys": len(_key_hits)}
