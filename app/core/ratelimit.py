import time
from collections import defaultdict, deque

WINDOW_SECONDS = 60
MAX_PER_KEY = 120  # keep low for verification; we'll raise later

_key_hits = defaultdict(deque)

def _prune(q: deque, now: float):
    cutoff = now - WINDOW_SECONDS
    while q and q[0] < cutoff:
        q.popleft()

def check_and_record(key: str) -> bool:
    now = time.time()
    q = _key_hits[key]
    _prune(q, now)
    if len(q) >= MAX_PER_KEY:
        return False
    q.append(now)
    return True

def debug_state():
    return {"key_keys": len(_key_hits)}
