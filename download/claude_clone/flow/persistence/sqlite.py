"""
SQLite persistence backend for flows.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flow.persistence.base import BasePersistenceBackend

logger = logging.getLogger(__name__)


class SQLitePersistenceBackend(BasePersistenceBackend):
    """
    SQLite-based persistence backend for flow state.

    Stores flow execution state in a local SQLite database file.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str = ".flow_state.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS flow_state (
                flow_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def save(self, flow_id: str, state: Dict[str, Any]) -> None:
        """Save or update flow execution state."""
        if not self._conn:
            return
        now = datetime.now().isoformat()
        state_json = json.dumps(state, default=str)
        self._conn.execute(
            """
            INSERT INTO flow_state (flow_id, state_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(flow_id) DO UPDATE SET
                state_json = excluded.state_json,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (flow_id, state_json, state.get("status", "running"), now, now),
        )
        self._conn.commit()
        logger.debug("Saved flow state for %s", flow_id)

    def load(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Load flow execution state."""
        if not self._conn:
            return None
        cursor = self._conn.execute(
            "SELECT state_json FROM flow_state WHERE flow_id = ?",
            (flow_id,),
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def delete(self, flow_id: str) -> bool:
        """Delete flow execution state."""
        if not self._conn:
            return False
        cursor = self._conn.execute(
            "DELETE FROM flow_state WHERE flow_id = ?",
            (flow_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def list_flows(self) -> List[Dict[str, Any]]:
        """List all persisted flows."""
        if not self._conn:
            return []
        cursor = self._conn.execute(
            "SELECT flow_id, status, created_at, updated_at FROM flow_state ORDER BY updated_at DESC"
        )
        return [
            {"flow_id": row[0], "status": row[1], "created_at": row[2], "updated_at": row[3]}
            for row in cursor.fetchall()
        ]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()
