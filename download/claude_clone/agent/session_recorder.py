"""
Session Recording & Replay system.

Records every agent action (user messages, tool calls, tool results, LLM responses,
errors, file changes, timing) into SQLite. Supports replay with fast-forward, export
to JSON/Markdown/HTML, full-text search, session comparison, and per-session statistics.

All database operations are async via ``sqlite3`` + ``asyncio.run_in_executor`` so
that no external async database drivers are required.

Usage::

    recorder = SessionRecorder()
    await recorder.initialize()
    session = await recorder.start_session("debug-auth-flow")
    await recorder.record_event(session.id, "user_message", {"content": "Fix auth bug"})
    # ... agent runs ...
    await recorder.stop_session(session.id)
    stats = await recorder.get_stats(session.id)
    export = await recorder.export(session.id, format="markdown")
"""

from __future__ import annotations

import asyncio
import html
import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "~/.claude_clone/sessions.db"

# Valid event types that can be recorded.
EVENT_TYPES = frozenset({
    "user_message", "llm_response", "tool_call", "tool_result",
    "error", "file_change", "thinking", "system", "done", "usage",
})

# Risk levels for exported HTML colouring.
_RISK_COLOURS = {
    "user_message": "#e3f2fd",
    "llm_response": "#e8f5e9",
    "tool_call": "#fff3e0",
    "tool_result": "#fce4ec",
    "error": "#ffebee",
    "file_change": "#f3e5f5",
    "thinking": "#f5f5f5",
    "system": "#eceff1",
    "done": "#e0f7fa",
    "usage": "#f9fbe7",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionEvent:
    """A single recorded event within a session."""

    id: str
    session_id: str
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    duration_ms: int = 0
    tokens: int = 0
    cost: float = 0.0


@dataclass
class Session:
    """A recorded session with its events."""

    id: str
    name: str
    start_time: str
    end_time: str = ""
    events: List[SessionEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _generate_id() -> str:
    return uuid.uuid4().hex[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_event(row: sqlite3.Row) -> SessionEvent:
    data = json.loads(row["data"]) if row["data"] else {}
    return SessionEvent(
        id=row["id"],
        session_id=row["session_id"],
        event_type=row["event_type"],
        data=data,
        timestamp=row["timestamp"],
        duration_ms=row["duration_ms"] or 0,
        tokens=row["tokens"] or 0,
        cost=row["cost"] or 0.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SessionRecorder
# ──────────────────────────────────────────────────────────────────────────────

class SessionRecorder:
    """
    Persistent session recording, replay, search, and export engine backed by SQLite.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. ``~`` is expanded automatically.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self._conn: Optional[sqlite3.Connection] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Create database schema and prepare the connection."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_sync(self._connect)
        await self._run_sync(self._create_tables)

    async def close(self) -> None:
        if self._conn is not None:
            await self._run_sync(self._conn.close)
            self._conn = None

    def _connect(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                start_time  TEXT NOT NULL,
                end_time    TEXT DEFAULT '',
                metadata    TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS events (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                data        TEXT DEFAULT '{}',
                timestamp   TEXT NOT NULL,
                duration_ms INTEGER DEFAULT 0,
                tokens      INTEGER DEFAULT 0,
                cost        REAL DEFAULT 0.0,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_events_session
                ON events(session_id, timestamp);

            CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(event_type);

            CREATE INDEX IF NOT EXISTS idx_events_data
                ON events(data);
        """)
        self._conn.commit()

    # ── Async wrapper ─────────────────────────────────────────────────────

    async def _run_sync(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SessionRecorder is not initialized. Call `await recorder.initialize()` first.")
        return self._conn

    # ── Session management ────────────────────────────────────────────────

    async def start_session(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Session:
        """
        Create a new session and return it.

        Parameters
        ----------
        name:
            Human-readable session label.
        metadata:
            Optional key-value metadata attached to the session.

        Returns
        -------
        Session
            The newly-created session (with no events yet).
        """
        session_id = _generate_id()
        now = _now_iso()

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO sessions (id, name, start_time, metadata) VALUES (?, ?, ?, ?)",
                (session_id, name, now, json.dumps(metadata or {})),
            )
            conn.commit()

        await self._run_sync(_do)
        return Session(id=session_id, name=name, start_time=now, metadata=metadata or {})

    async def stop_session(self, session_id: str) -> None:
        """Mark a session as finished by setting ``end_time``."""
        now = _now_iso()

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "UPDATE sessions SET end_time = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()

        await self._run_sync(_do)

    # ── Event recording ───────────────────────────────────────────────────

    async def record_event(
        self,
        session_id: str,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        duration_ms: int = 0,
        tokens: int = 0,
        cost: float = 0.0,
    ) -> str:
        """
        Record a single event into a session.

        Parameters
        ----------
        session_id:
            Target session.
        event_type:
            One of ``EVENT_TYPES`` (e.g. ``"tool_call"``, ``"error"``).
        data:
            Arbitrary event payload dict.
        duration_ms:
            How long the event took in milliseconds.
        tokens:
            Number of tokens consumed (if applicable).
        cost:
            Dollar cost of the event (if applicable).

        Returns
        -------
        str
            The event id.
        """
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Invalid event_type: {event_type!r}. Must be one of {sorted(EVENT_TYPES)}")

        event_id = _generate_id()
        now = _now_iso()

        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO events "
                "(id, session_id, event_type, data, timestamp, duration_ms, tokens, cost) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (event_id, session_id, event_type, json.dumps(data or {}, default=str), now,
                 duration_ms, tokens, cost),
            )
            conn.commit()

        await self._run_sync(_do)
        return event_id

    # ── Replay ────────────────────────────────────────────────────────────

    async def replay(
        self,
        session_id: str,
        skip_to: Optional[str] = None,
        skip_to_time: Optional[str] = None,
    ) -> List[SessionEvent]:
        """
        Retrieve recorded events for replay.

        Parameters
        ----------
        session_id:
            Target session.
        skip_to:
            If given, start from the event with this id (inclusive).
        skip_to_time:
            If given, start from the first event at or after this ISO timestamp.

        Returns
        -------
        list[SessionEvent]
            Events in chronological order.
        """
        def _do() -> List[SessionEvent]:
            conn = self._ensure_conn()
            query = "SELECT * FROM events WHERE session_id = ?"
            params: List[Any] = [session_id]

            if skip_to is not None:
                query += " AND id >= ?"
                params.append(skip_to)
            elif skip_to_time is not None:
                query += " AND timestamp >= ?"
                params.append(skip_to_time)

            query += " ORDER BY timestamp ASC"
            rows = conn.execute(query, params).fetchall()
            return [_row_to_event(r) for r in rows]

        return await self._run_sync(_do)

    # ── Search ────────────────────────────────────────────────────────────

    async def search(
        self,
        session_id: Optional[str] = None,
        query: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[SessionEvent]:
        """
        Search events by text query and/or type.

        The text query is matched against the ``data`` JSON column using SQLite's
        LIKE operator, which is sufficient for finding file names, error messages,
        and tool names.

        Parameters
        ----------
        session_id:
            Restrict to a single session.
        query:
            Free-text substring to search for in event data.
        event_type:
            Filter to a specific event type.
        limit:
            Maximum results.

        Returns
        -------
        list[SessionEvent]
            Matching events sorted by timestamp descending.
        """
        def _do() -> List[SessionEvent]:
            conn = self._ensure_conn()
            clauses: List[str] = []
            params: List[Any] = []

            if session_id is not None:
                clauses.append("session_id = ?")
                params.append(session_id)
            if query is not None:
                clauses.append("data LIKE ?")
                params.append(f"%{query}%")
            if event_type is not None:
                clauses.append("event_type = ?")
                params.append(event_type)

            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = conn.execute(
                f"SELECT * FROM events{where} ORDER BY timestamp DESC LIMIT ?",
                params + [limit],
            ).fetchall()
            return [_row_to_event(r) for r in rows]

        return await self._run_sync(_do)

    # ── Statistics ────────────────────────────────────────────────────────

    async def get_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Compute statistics for a single session.

        Returns a dict with keys: ``duration_seconds``, ``total_tokens``,
        ``total_cost``, ``event_counts``, ``tools_called``, ``files_modified``,
        ``error_count``, ``events``.
        """
        def _do() -> Dict[str, Any]:
            conn = self._ensure_conn()
            session_row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if session_row is None:
                raise KeyError(f"Session '{session_id}' not found")

            events = conn.execute(
                "SELECT * FROM events WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            ).fetchall()

            start = session_row["start_time"]
            end = session_row["end_time"] or _now_iso()
            try:
                start_dt = datetime.fromisoformat(start)
                end_dt = datetime.fromisoformat(end)
                duration = (end_dt - start_dt).total_seconds()
            except (ValueError, TypeError):
                duration = 0.0

            total_tokens = sum(e["tokens"] or 0 for e in events)
            total_cost = sum(e["cost"] or 0.0 for e in events)

            event_counts: Dict[str, int] = {}
            tools_called: List[str] = []
            files_modified: List[str] = []
            error_count = 0

            for ev in events:
                etype = ev["event_type"]
                event_counts[etype] = event_counts.get(etype, 0) + 1

                data = json.loads(ev["data"]) if ev["data"] else {}
                if etype == "tool_call":
                    name = data.get("name", "unknown")
                    if name not in tools_called:
                        tools_called.append(name)
                if etype == "file_change":
                    path = data.get("path", data.get("file", "unknown"))
                    if path not in files_modified:
                        files_modified.append(path)
                if etype == "error":
                    error_count += 1

            return {
                "session_id": session_id,
                "name": session_row["name"],
                "duration_seconds": round(duration, 2),
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 6),
                "event_count": len(events),
                "event_counts": event_counts,
                "tools_called": tools_called,
                "files_modified": files_modified,
                "error_count": error_count,
                "start_time": start,
                "end_time": end,
            }

        return await self._run_sync(_do)

    # ── List sessions ─────────────────────────────────────────────────────

    async def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return metadata for all sessions, most-recent first."""
        def _do() -> List[Dict[str, Any]]:
            conn = self._ensure_conn()
            rows = conn.execute("""
                SELECT s.id, s.name, s.start_time, s.end_time, s.metadata,
                       COUNT(e.id) AS event_count
                FROM sessions s
                LEFT JOIN events e ON e.session_id = s.id
                GROUP BY s.id
                ORDER BY s.start_time DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                    "event_count": r["event_count"],
                }
                for r in rows
            ]

        return await self._run_sync(_do)

    # ── Session comparison ────────────────────────────────────────────────

    async def compare_sessions(self, session_id_a: str, session_id_b: str) -> Dict[str, Any]:
        """
        Compare two sessions side by side.

        Returns a dict with ``session_a_stats``, ``session_b_stats``,
        ``differences`` (list of dicts describing what differs), and a
        ``summary`` string.
        """
        stats_a = await self.get_stats(session_id_a)
        stats_b = await self.get_stats(session_id_b)

        differences: List[Dict[str, Any]] = []

        # Compare event counts by type.
        all_types = sorted(set(list(stats_a["event_counts"].keys()) + list(stats_b["event_counts"].keys())))
        for etype in all_types:
            count_a = stats_a["event_counts"].get(etype, 0)
            count_b = stats_b["event_counts"].get(etype, 0)
            if count_a != count_b:
                differences.append({
                    "field": f"event_count:{etype}",
                    "session_a": count_a,
                    "session_b": count_b,
                })

        # Compare tools called.
        tools_a = set(stats_a["tools_called"])
        tools_b = set(stats_b["tools_called"])
        if tools_a != tools_b:
            differences.append({
                "field": "tools_called",
                "only_in_a": sorted(tools_a - tools_b),
                "only_in_b": sorted(tools_b - tools_a),
            })

        # Compare files modified.
        files_a = set(stats_a["files_modified"])
        files_b = set(stats_b["files_modified"])
        if files_a != files_b:
            differences.append({
                "field": "files_modified",
                "only_in_a": sorted(files_a - files_b),
                "only_in_b": sorted(files_b - files_a),
            })

        # Compare token / cost / duration.
        for key in ("total_tokens", "total_cost", "duration_seconds", "error_count"):
            if stats_a[key] != stats_b[key]:
                differences.append({
                    "field": key,
                    "session_a": stats_a[key],
                    "session_b": stats_b[key],
                })

        summary = (
            f"Compared '{stats_a['name']}' ({stats_a['event_count']} events) vs "
            f"'{stats_b['name']}' ({stats_b['event_count']} events). "
            f"Found {len(differences)} difference(s)."
        )

        return {
            "session_a": stats_a,
            "session_b": stats_b,
            "differences": differences,
            "summary": summary,
        }

    # ── Export ────────────────────────────────────────────────────────────

    async def export(
        self,
        session_id: str,
        format: str = "json",  # noqa: A002 — intentional parameter name
        filepath: Optional[str] = None,
    ) -> str:
        """
        Export a session to a chosen format.

        Parameters
        ----------
        session_id:
            Target session.
        format:
            ``"json"``, ``"markdown"``, or ``"html"``.
        filepath:
            If given, also write the output to this file.

        Returns
        -------
        str
            The exported content.
        """
        stats = await self.get_stats(session_id)
        events = await self.replay(session_id)

        if format == "json":
            content = self._export_json(stats, events)
        elif format == "markdown":
            content = self._export_markdown(stats, events)
        elif format == "html":
            content = self._export_html(stats, events)
        else:
            raise ValueError(f"Unknown export format: {format!r}. Use json, markdown, or html.")

        if filepath is not None:
            path = Path(filepath).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)

            def _write() -> None:
                path.write_text(content, encoding="utf-8")

            await self._run_sync(_write)

        return content

    @staticmethod
    def _export_json(stats: Dict[str, Any], events: List[SessionEvent]) -> str:
        payload = {
            "session": {
                "id": stats["session_id"],
                "name": stats["name"],
                "start_time": stats["start_time"],
                "end_time": stats["end_time"],
            },
            "statistics": {
                "duration_seconds": stats["duration_seconds"],
                "total_tokens": stats["total_tokens"],
                "total_cost": stats["total_cost"],
                "event_count": stats["event_count"],
                "tools_called": stats["tools_called"],
                "files_modified": stats["files_modified"],
                "error_count": stats["error_count"],
            },
            "events": [asdict(e) for e in events],
        }
        return json.dumps(payload, indent=2, default=str, ensure_ascii=False)

    @staticmethod
    def _export_markdown(stats: Dict[str, Any], events: List[SessionEvent]) -> str:
        lines: List[str] = [
            f"# Session: {stats['name']}",
            "",
            f"**ID:** `{stats['session_id']}`",
            f"**Start:** {stats['start_time']}",
            f"**End:** {stats['end_time']}",
            f"**Duration:** {stats['duration_seconds']}s",
            f"**Tokens:** {stats['total_tokens']}",
            f"**Cost:** ${stats['total_cost']:.4f}",
            f"**Events:** {stats['event_count']}",
            f"**Errors:** {stats['error_count']}",
            "",
        ]

        if stats["tools_called"]:
            lines.append(f"**Tools:** {', '.join(f'`{t}`' for t in stats['tools_called'])}")
            lines.append("")

        if stats["files_modified"]:
            lines.append(f"**Files modified:** {', '.join(f'`{f}`' for f in stats['files_modified'])}")
            lines.append("")

        lines.append("---")
        lines.append("")

        for ev in events:
            ts = ev.timestamp[:19] if ev.timestamp else "unknown"
            lines.append(f"### [{ts}] {ev.event_type}")
            if ev.duration_ms:
                lines.append(f"**Duration:** {ev.duration_ms}ms")
            if ev.tokens:
                lines.append(f"**Tokens:** {ev.tokens}")
            if ev.cost:
                lines.append(f"**Cost:** ${ev.cost:.4f}")

            if ev.data:
                data_str = json.dumps(ev.data, indent=2, default=str, ensure_ascii=False)
                if len(data_str) > 500:
                    data_str = data_str[:500] + "\n... (truncated)"
                lines.append(f"```json\n{data_str}\n```")

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _export_html(stats: Dict[str, Any], events: List[SessionEvent]) -> str:
        event_rows: List[str] = []
        for ev in events:
            bg = _RISK_COLOURS.get(ev.event_type, "#ffffff")
            ts = html.escape(ev.timestamp[:19] if ev.timestamp else "unknown")
            data_json = html.escape(json.dumps(ev.data, indent=2, default=str, ensure_ascii=False)[:2000])
            event_rows.append(
                f'<tr style="background:{bg}">'
                f"<td>{ts}</td>"
                f"<td><code>{html.escape(ev.event_type)}</code></td>"
                f"<td>{ev.duration_ms}ms</td>"
                f"<td>{ev.tokens}</td>"
                f"<td>${ev.cost:.4f}</td>"
                f"<td><pre>{data_json}</pre></td>"
                f"</tr>"
            )

        return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Session: {html.escape(stats['name'])}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #fafafa; }}
  h1 {{ color: #333; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
  th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; vertical-align: top; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 0.8rem; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0; }}
  .stat-card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; }}
  .stat-card h3 {{ margin: 0 0 0.25rem; font-size: 0.85rem; color: #666; }}
  .stat-card p {{ margin: 0; font-size: 1.1rem; font-weight: 600; }}
</style>
</head>
<body>
<h1>Session: {html.escape(stats['name'])}</h1>
<div class="stats">
  <div class="stat-card"><h3>Duration</h3><p>{stats['duration_seconds']}s</p></div>
  <div class="stat-card"><h3>Tokens</h3><p>{stats['total_tokens']:,}</p></div>
  <div class="stat-card"><h3>Cost</h3><p>${stats['total_cost']:.4f}</p></div>
  <div class="stat-card"><h3>Events</h3><p>{stats['event_count']}</p></div>
  <div class="stat-card"><h3>Errors</h3><p>{stats['error_count']}</p></div>
  <div class="stat-card"><h3>Tools</h3><p>{', '.join(stats['tools_called']) or 'None'}</p></div>
</div>
<table>
<tr><th>Timestamp</th><th>Type</th><th>Duration</th><th>Tokens</th><th>Cost</th><th>Data</th></tr>
{"".join(event_rows)}
</table>
</body>
</html>"""

    # ── Delete ────────────────────────────────────────────────────────────

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and all its events."""
        def _do() -> None:
            conn = self._ensure_conn()
            conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()

        await self._run_sync(_do)
