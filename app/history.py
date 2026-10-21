"""Chat history that survives page reloads and server restarts.

One SQLite file (data/history.db by default). Each browser keeps a random
visitor id, and history is stored per visitor per portrait.
"""
import re
import sqlite3
import threading
import time
from pathlib import Path

_VISITOR_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def valid_visitor(visitor: str | None) -> bool:
    return bool(visitor and _VISITOR_ID.match(visitor))


class HistoryStore:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.db:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor TEXT NOT NULL,
                    persona TEXT NOT NULL,
                    role TEXT NOT NULL,          -- "user" or "assistant"
                    content TEXT NOT NULL,
                    created REAL NOT NULL
                )""")
            self.db.execute("CREATE INDEX IF NOT EXISTS by_chat ON messages (visitor, persona, id)")

    def add(self, visitor: str, persona: str, role: str, content: str):
        if not content:
            return
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO messages (visitor, persona, role, content, created) VALUES (?, ?, ?, ?, ?)",
                (visitor, persona, role, content, time.time()))

    def recent(self, visitor: str, persona: str, limit: int = 50) -> list[dict]:
        """The last `limit` messages, oldest first."""
        with self.lock:
            rows = self.db.execute(
                "SELECT role, content FROM messages WHERE visitor = ? AND persona = ? "
                "ORDER BY id DESC LIMIT ?", (visitor, persona, limit)).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def clear(self, visitor: str, persona: str):
        with self.lock, self.db:
            self.db.execute("DELETE FROM messages WHERE visitor = ? AND persona = ?", (visitor, persona))
