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
    done INTEGER NOT NULL DEFAULT 0
);
-- Places paid for on Apify, per local day.
CREATE TABLE IF NOT EXISTS usage (
    day TEXT PRIMARY KEY,
    places INTEGER NOT NULL
);
-- Places bought on Apify but not processed yet: nothing paid for is thrown away.
CREATE TABLE IF NOT EXISTS pending (
    position INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id TEXT NOT NULL UNIQUE,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id TEXT NOT NULL UNIQUE,
    -- NULL when the business has no Instagram (contact by phone / WhatsApp).
    instagram TEXT UNIQUE,
    context TEXT NOT NULL,
    score INTEGER NOT NULL,
    reason TEXT NOT NULL,
    idea TEXT NOT NULL,
    message TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
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
        self._migrate()

    def _migrate(self) -> None:
        columns = {r["name"]: r for r in self.conn.execute("PRAGMA table_info(leads)")}
        if "summary" not in columns:
            self.conn.execute("ALTER TABLE leads ADD COLUMN summary TEXT NOT NULL DEFAULT ''")
        if columns["instagram"]["notnull"]:
            # Leads without Instagram are allowed now; SQLite can only drop NOT NULL by rebuilding the table.
            with self.conn:
                self.conn.execute("ALTER TABLE leads RENAME TO leads_old")
                self.conn.executescript(SCHEMA)
                cols = ", ".join(r["name"] for r in self.conn.execute("PRAGMA table_info(leads_old)"))
                self.conn.execute(f"INSERT INTO leads({cols}) SELECT {cols} FROM leads_old")
                self.conn.execute("DROP TABLE leads_old")

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

    def next_queries(self, n: int) -> list[str]:
        rows = self.conn.execute(
            "SELECT query FROM queries WHERE done = 0 ORDER BY position LIMIT ?", (n,)
        ).fetchall()
        return [r["query"] for r in rows]

    def finish_queries(self, queries: list[str]) -> None:
        with self.conn:
            self.conn.executemany("UPDATE queries SET done = 1 WHERE query = ?", [(q,) for q in queries])

    def restart_queries(self) -> None:
        """All searches are done: start over to pick up businesses that appeared on Maps since."""
        with self.conn:
            self.conn.execute("UPDATE queries SET done = 0")

    # ---------- Apify budget ----------

    def places_bought(self, day: str) -> int:
        row = self.conn.execute("SELECT places FROM usage WHERE day = ?", (day,)).fetchone()
        return row["places"] if row else 0

    def add_places_bought(self, day: str, n: int) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO usage(day, places) VALUES (?, ?) ON CONFLICT(day) DO UPDATE SET places = places + ?",
                (day, n, n),
            )

    # ---------- bought, not yet processed ----------

    def add_pending(self, items: list[tuple[str, dict]]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO pending(place_id, data) VALUES (?, ?)",
                [(pid, json.dumps(data, ensure_ascii=False)) for pid, data in items],
            )

    def pop_pending(self) -> dict | None:
        row = self.conn.execute("SELECT position, data FROM pending ORDER BY position LIMIT 1").fetchone()
        if not row:
            return None
        with self.conn:
            self.conn.execute("DELETE FROM pending WHERE position = ?", (row["position"],))
        return json.loads(row["data"])

    def recover(self, items: list[tuple[str, dict]]) -> int:
        """Queues places bought earlier that never became a card (older versions dropped them). Returns how many."""
        added = 0
        with self.conn:
            for pid, data in items:
                if self.conn.execute("SELECT 1 FROM leads WHERE place_id = ?", (pid,)).fetchone():
                    continue
                self.conn.execute("DELETE FROM places WHERE id = ?", (pid,))
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO pending(place_id, data) VALUES (?, ?)",
                    (pid, json.dumps(data, ensure_ascii=False)),
                )
                added += cur.rowcount
        return added

    def pending_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM pending").fetchone()[0]

    # ---------- places ----------

    def place_checked(self, place_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM places WHERE id = ?", (place_id,)).fetchone() is not None

    def mark_place(self, place_id: str, status: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO places(id, status, checked_at) VALUES (?, ?, ?)",
                (place_id, status, _now()),
            )

    def instagram_taken(self, handle: str | None) -> bool:
        if not handle:
            return False
        return self.conn.execute("SELECT 1 FROM leads WHERE instagram = ?", (handle,)).fetchone() is not None

    # ---------- leads ----------

    def add_lead(self, place_id: str, instagram: str | None, context: dict, score: int, reason: str, idea: str,
                 message: str, summary: str = "") -> int:
        now = _now()
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO leads(place_id, instagram, context, score, reason, idea, message, summary, created_at,"
                " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (place_id, instagram, json.dumps(context, ensure_ascii=False), score, reason, idea, message, summary,
                 now, now),
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
