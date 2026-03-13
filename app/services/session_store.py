"""In-memory conversation session store with TTL eviction and history trimming."""
import time
from typing import List

# Sessions older than this are evicted
SESSION_TTL_SECONDS = 1800  # 30 minutes
# Keep at most this many messages per session (prevents context overflow)
MAX_HISTORY_MESSAGES = 40


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, dict] = {}

    def get(self, session_id: str) -> List[dict]:
        self._evict_stale()
        entry = self._sessions.get(session_id)
        if entry is None:
            return []
        entry["last_access"] = time.time()
        return entry["messages"]

    def set(self, session_id: str, messages: List[dict]) -> None:
        # Trim to max length, keeping the most recent messages
        if len(messages) > MAX_HISTORY_MESSAGES:
            messages = messages[-MAX_HISTORY_MESSAGES:]
        self._sessions[session_id] = {
            "messages": messages,
            "last_access": time.time(),
        }

    def _evict_stale(self) -> None:
        cutoff = time.time() - SESSION_TTL_SECONDS
        stale = [sid for sid, v in self._sessions.items() if v["last_access"] < cutoff]
        for sid in stale:
            del self._sessions[sid]


session_store = SessionStore()
