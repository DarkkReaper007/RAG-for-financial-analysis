"""API key pool: use the first key; when one is rate-limited (429) or rejected (401/403) cool it down and fall over to the next.
Keys come from the environment only: <NAME>_KEYS (comma separated) or <NAME>_KEY. Never log or persist keys."""
import os
import threading
import time

_pools = {}


class KeyPool:
    def __init__(self, keys):
        self.keys = [k.strip() for k in keys if k.strip()]
        self.cool_until = [0.0] * len(self.keys)
        self.lock = threading.Lock()

    def get(self):
        """-> (index, key). The first key that is not cooling; if all are cooling, the one that frees up soonest."""
        with self.lock:
            now = time.time()
            for i, u in enumerate(self.cool_until):
                if u <= now:
                    return i, self.keys[i]
            i = min(range(len(self.keys)), key=lambda j: self.cool_until[j])
            return i, self.keys[i]

    def cool(self, i, seconds):
        with self.lock:
            self.cool_until[i] = max(self.cool_until[i], time.time() + seconds)

    def wait_needed(self):
        """Seconds until some key is usable (0 if one is available now)."""
        with self.lock:
            return max(0.0, min(self.cool_until) - time.time()) if self.keys else 0.0


def pool(env_single):
    """Pool for an env var such as GROQ_API_KEY (reads GROQ_API_KEYS first, then GROQ_API_KEY)."""
    if env_single not in _pools:
        raw = os.environ.get(env_single.replace("_KEY", "_KEYS")) or os.environ.get(env_single) or ""
        _pools[env_single] = KeyPool(raw.split(","))
    return _pools[env_single]


def post_with_pool(requests_post, env_single, url, payload, timeout, max_attempts=10):
    """POST with key fail-over. Returns the last response (or raises the last request exception)."""
    p = pool(env_single)
    if not p.keys:
        raise KeyError(f"no API key in {env_single}/{env_single.replace('_KEY', '_KEYS')}")
    r = None
    for attempt in range(max_attempts):
        w = p.wait_needed()
        if w > 0:
            time.sleep(min(w, 60))
        i, key = p.get()
        r = requests_post(url, headers={"Authorization": "Bearer " + key}, json=payload, timeout=timeout)
        if r.status_code == 429:
            p.cool(i, min(120.0, float(r.headers.get("retry-after") or 0) or 15.0))
            continue
        if r.status_code in (401, 403):
            p.cool(i, 3600.0)                      # rejected key: sideline for an hour
            continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(min(30, 2 ** attempt))
            continue
        return r
    return r
