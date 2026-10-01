"""SQLite storage: checked places, search progress and leads."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS places (
    id TEXT PRIMARY KEY,
    -- Why the place was dropped, or 'lead'.
    status TEXT NOT NULL,
    checked_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS queries (
    query TEXT PRIMARY KEY,
    position INTEGER NOT NULL,
    page_token TEXT,
    exhausted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id TEXT NOT NULL UNIQUE,
    instagram TEXT NOT NULL UNIQUE,
    context TEXT NOT NULL,
    score INTEGER NOT NULL,
    reason TEXT NOT NULL,
    idea TEXT NOT NULL,
    message TEXT NOT NULL,
    -- new / sent / rejected
    status TEXT NOT NULL DEFAULT 'new',
    handled_by TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DB:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ---------- search progress ----------

    def sync_queries(self, queries: list[str]) -> None:
        """Adds new queries; existing ones keep their progress. Order follows the list."""
        with self.conn:
            for i, q in enumerate(queries):
                self.conn.execute(
                    "INSERT INTO queries(position, query) VALUES (?, ?) "
                    "ON CONFLICT(query) DO UPDATE SET position = excluded.position",
                    (i, q),
                )
            self.conn.execute(
                f"DELETE FROM queries WHERE query NOT IN ({','.join('?' * len(queries))})", queries
            )

    def next_query(self) -> tuple[str, str | None] | None:
        row = self.conn.execute(
            "SELECT query, page_token FROM queries WHERE exhausted = 0 ORDER BY position LIMIT 1"
        ).fetchone()
        return (row["query"], row["page_token"]) if row else None

    def advance_query(self, query: str, next_token: str | None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE queries SET page_token = ?, exhausted = ? WHERE query = ?",
                (next_token, int(next_token is None), query),
            )

    def restart_queries(self) -> None:
        """All searches are done: start over to pick up businesses that appeared on Maps since."""
        with self.conn:
            self.conn.execute("UPDATE queries SET page_token = NULL, exhausted = 0")

    # ---------- places ----------

    def place_checked(self, place_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM places WHERE id = ?", (place_id,)).fetchone() is not None

    def mark_place(self, place_id: str, status: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO places(id, status, checked_at) VALUES (?, ?, ?)",
                (place_id, status, _now()),
            )

    def instagram_taken(self, handle: str) -> bool:
        return self.conn.execute("SELECT 1 FROM leads WHERE instagram = ?", (handle,)).fetchone() is not None

    # ---------- leads ----------

    def add_lead(self, place_id: str, instagram: str, context: dict, score: int, reason: str, idea: str,
                 message: str) -> int:
        now = _now()
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO leads(place_id, instagram, context, score, reason, idea, message, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (place_id, instagram, json.dumps(context, ensure_ascii=False), score, reason, idea, message, now,
                 now),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO places(id, status, checked_at) VALUES (?, 'lead', ?)", (place_id, now)
            )
        return cur.lastrowid

    def lead(self, lead_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
        if not row:
            return None
        lead = dict(row)
        lead["context"] = json.loads(lead["context"])
        return lead

    def set_status(self, lead_id: int, status: str, handled_by: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE leads SET status = ?, handled_by = ?, updated_at = ? WHERE id = ?",
                (status, handled_by, _now(), lead_id),
            )

    def set_message(self, lead_id: int, message: str, idea: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE leads SET message = ?, idea = ?, updated_at = ? WHERE id = ?",
                (message, idea, _now(), lead_id),
            )

    # ---------- stats ----------

    def stats(self, since: str) -> dict:
        leads = dict(self.conn.execute("SELECT status, COUNT(*) FROM leads GROUP BY status").fetchall())
        places = dict(self.conn.execute("SELECT status, COUNT(*) FROM places GROUP BY status").fetchall())
        today = self.conn.execute("SELECT COUNT(*) FROM leads WHERE created_at >= ?", (since,)).fetchone()[0]
        sent_today = self.conn.execute(
            "SELECT COUNT(*) FROM leads WHERE status = 'sent' AND updated_at >= ?", (since,)
        ).fetchone()[0]
        return {"leads": leads, "places": places, "today": today, "sent_today": sent_today}
