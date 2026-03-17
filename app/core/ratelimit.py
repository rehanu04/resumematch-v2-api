import threading
import time
from collections import deque

WINDOW_SECONDS = 60
MAX_PER_KEY = 120  # keep low for verification; we'll raise later

_key_hits: dict[str, deque] = {}
_lock = threading.Lock()

def _prune(q: deque, now: float):
    cutoff = now - WINDOW_SECONDS
    while q and q[0] < cutoff:
        q.popleft()

def check_and_record(key: str) -> bool:
    now = time.time()
    with _lock:
        q = _key_hits.get(key)
        if q is not None:
            _prune(q, now)
            if not q:
                # All entries expired; remove stale key to prevent memory leak
                del _key_hits[key]
                q = None
        if q is None:
            q = deque()
        if len(q) >= MAX_PER_KEY:
            return False
        q.append(now)
        _key_hits[key] = q
        return True

def debug_state():
    with _lock:
        return {"key_keys": len(_key_hits)}
