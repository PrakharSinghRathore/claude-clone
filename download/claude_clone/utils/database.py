"""
Database Manager supporting SQLite and PostgreSQL.

Provides a unified async interface for common database operations including
CRUD, schema introspection, CSV import/export, migrations, and transactions.

Usage:
    db = DatabaseManager("sqlite:///mydb.db")
    await db.connect()
    rows = await db.execute("SELECT * FROM users WHERE id = %s", (1,))
    await db.disconnect()
"""

from __future__ import annotations

import csv
import logging
import os
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Async database manager with SQLite and PostgreSQL support."""

    def __init__(self, connection_string: Optional[str] = None) -> None:
        """Initialize the database manager.

        Args:
            connection_string: Database URI. SQLite format: ``sqlite:///path/to/db.db``
                or just a filename. PostgreSQL format: ``postgresql://user:pass@host:port/dbname``.
                When *None*, defaults to an in-memory SQLite database.
        """
        self.connection_string = connection_string or "sqlite:///:memory:"
        self.db_type: str = "sqlite"
        self._connection: Any = None
        self._cursor: Any = None
        self._sqlite_path: Optional[str] = None
        self._in_transaction: bool = False

        self._parse_connection_string()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _parse_connection_string(self) -> None:
        """Detect database type and extract relevant parameters."""
        if self.connection_string.startswith("postgresql://") or self.connection_string.startswith("postgres://"):
            self.db_type = "postgresql"
            return

        if self.connection_string.startswith("sqlite:///"):
            self._sqlite_path = self.connection_string[len("sqlite:///"):]
            self.db_type = "sqlite"
        elif not self.connection_string.startswith("sqlite://"):
            # Treat bare path as SQLite file
            self._sqlite_path = self.connection_string
            self.db_type = "sqlite"

    async def connect(self) -> None:
        """Open a database connection."""
        if self.db_type == "sqlite":
            self._connect_sqlite()
        else:
            await self._connect_postgresql()
        logger.info("Connected to %s database.", self.db_type)

    def _connect_sqlite(self) -> None:
        """Open a synchronous SQLite connection (wrapped for async interface)."""
        detect = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        if self._sqlite_path == ":memory:" or self._sqlite_path is None:
            self._connection = sqlite3.connect(":memory:", detect_types=detect)
        else:
            parent = os.path.dirname(self._sqlite_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._connection = sqlite3.connect(self._sqlite_path, detect_types=detect)
        self._connection.row_factory = sqlite3.Row
        self._cursor = self._connection.cursor()

    async def _connect_postgresql(self) -> None:
        """Open an asynchronous PostgreSQL connection using psycopg2 (sync fallback)."""
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError as exc:
            raise ImportError(
                "psycopg2 is required for PostgreSQL support. "
                "Install it with: pip install psycopg2-binary"
            ) from exc

        self._connection = psycopg2.connect(self.connection_string, cursor_factory=psycopg2.extras.RealDictCursor)
        self._connection.autocommit = True
        self._cursor = self._connection.cursor()

    async def disconnect(self) -> None:
        """Close the database connection."""
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._connection:
            self._connection.close()
            self._connection = None
        logger.info("Disconnected from %s database.", self.db_type)

    # ------------------------------------------------------------------
    # Core query interface
    # ------------------------------------------------------------------

    async def execute(self, query: str, params: Optional[tuple] = None) -> list[dict]:
        """Execute a query and return matching rows as a list of dicts.

        Args:
            query: SQL query string.
            params: Optional bind parameters.

        Returns:
            List of row dictionaries. For DML statements returns an empty list.
        """
        await self._ensure_connected()
        logger.debug("execute: %s | params=%s", query, params)

        try:
            self._cursor.execute(query, params or ())
            if self._cursor.description:
                columns = [desc[0] for desc in self._cursor.description]
                rows = self._cursor.fetchall()
                if self.db_type == "sqlite":
                    return [dict(zip(columns, row)) for row in rows]
                return [dict(row) for row in rows]
            if self.db_type == "sqlite":
                self._connection.commit()
            return []
        except Exception as exc:
            logger.error("Query failed: %s — %s", query, exc)
            raise

    async def execute_many(self, query: str, params_list: list[tuple]) -> int:
        """Execute *query* for every parameter set in *params_list*.

        Returns:
            Total number of affected rows.
        """
        await self._ensure_connected()
        if not params_list:
            return 0

        total = 0
        try:
            if self.db_type == "sqlite":
                self._cursor.executemany(query, params_list)
                total = self._cursor.rowcount
                self._connection.commit()
            else:
                for params in params_list:
                    self._cursor.execute(query, params)
                    total += self._cursor.rowcount or 0
            logger.debug("execute_many affected %d rows.", total)
            return total
        except Exception as exc:
            logger.error("execute_many failed: %s — %s", query, exc)
            raise

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    async def get_tables(self) -> list[dict]:
        """Return a list of tables with their approximate row counts.

        Returns:
            List of dicts with keys ``name`` and ``row_count``.
        """
        if self.db_type == "sqlite":
            tables = await self.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        else:
            tables = await self.execute(
                "SELECT table_name AS name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY table_name"
            )

        result: list[dict] = []
        for tbl in tables:
            name = tbl["name"]
            if self.db_type == "sqlite":
                count_rows = await self.execute(f'SELECT COUNT(*) AS cnt FROM "{name}"')
            else:
                count_rows = await self.execute(f'SELECT COUNT(*) AS cnt FROM "{name}"')
            row_count = count_rows[0]["cnt"] if count_rows else 0
            result.append({"name": name, "row_count": row_count})
        return result

    async def get_schema(self, table_name: str) -> list[dict]:
        """Return column metadata for *table_name*.

        Each dict contains ``name``, ``type``, ``nullable``, ``default``, and
        ``primary_key``.
        """
        if self.db_type == "sqlite":
            rows = await self.execute(f'PRAGMA table_info("{table_name}")')
            return [
                {
                    "name": r["name"],
                    "type": r["type"],
                    "nullable": r["notnull"] == 0,
                    "default": r["dflt_value"],
                    "primary_key": r["pk"] > 0,
                }
                for r in rows
            ]
        else:
            rows = await self.execute(
                "SELECT column_name AS name, data_type AS type, "
                "is_nullable AS nullable, column_default AS \"default\" "
                "FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=%s "
                "ORDER BY ordinal_position",
                (table_name,),
            )
            pk_rows = await self.execute(
                "SELECT a.attname AS name "
                "FROM pg_index i "
                "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                "WHERE i.indrelid = %s::regclass AND i.indisprimary",
                (table_name,),
            )
            pk_names = {r["name"] for r in pk_rows}
            for col in rows:
                col["nullable"] = col["nullable"] == "YES"
                col["primary_key"] = col["name"] in pk_names
            return rows

    async def get_table_data(
        self, table_name: str, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        """Return rows from *table_name* paginated by *limit* and *offset*."""
        rows = await self.execute(
            f'SELECT * FROM "{table_name}" LIMIT {int(limit)} OFFSET {int(offset)}'
        )
        return rows

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------

    async def insert(self, table: str, data: dict) -> int:
        """Insert a single row into *table*.

        Returns:
            The ``lastrowid`` (SQLite) or 1 if successful (PostgreSQL).
        """
        columns = list(data.keys())
        placeholders = ", ".join(["%s"] * len(columns))
        col_str = ", ".join(f'"{c}"' for c in columns)
        query = f'INSERT INTO "{table}" ({col_str}) VALUES ({placeholders})'
        values = tuple(data[c] for c in columns)

        await self._ensure_connected()
        self._cursor.execute(query, values)

        if self.db_type == "sqlite":
            self._connection.commit()
            return self._cursor.lastrowid or 0

        self._connection.commit()
        return 1

    async def update(
        self,
        table: str,
        data: dict,
        where: str,
        params: Optional[tuple] = None,
    ) -> int:
        """Update rows in *table* matching *where* clause.

        Returns:
            Number of affected rows.
        """
        set_clause = ", ".join(f'"{k}" = %s' for k in data.keys())
        query = f'UPDATE "{table}" SET {set_clause} WHERE {where}'
        values = tuple(data.values()) + (params or ())

        await self._ensure_connected()
        self._cursor.execute(query, values)

        if self.db_type == "sqlite":
            self._connection.commit()

        return self._cursor.rowcount or 0

    async def delete(
        self,
        table: str,
        where: str,
        params: Optional[tuple] = None,
    ) -> int:
        """Delete rows from *table* matching *where* clause.

        Returns:
            Number of deleted rows.
        """
        query = f'DELETE FROM "{table}" WHERE {where}'
        await self._ensure_connected()
        self._cursor.execute(query, params or ())

        if self.db_type == "sqlite":
            self._connection.commit()

        return self._cursor.rowcount or 0

    # ------------------------------------------------------------------
    # CSV import / export
    # ------------------------------------------------------------------

    async def export_to_csv(self, table: str, filepath: str) -> int:
        """Export all rows from *table* to a CSV file.

        Returns:
            Number of exported rows.
        """
        rows = await self.execute(f'SELECT * FROM "{table}"')
        if not rows:
            return 0

        columns = list(rows[0].keys())
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Exported %d rows from %s to %s", len(rows), table, filepath)
        return len(rows)

    async def import_from_csv(self, table: str, filepath: str) -> int:
        """Import rows from a CSV file into *table*.

        Returns:
            Number of imported rows.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        with open(filepath, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            columns = reader.fieldnames or []
            if not columns:
                return 0
            rows = list(reader)

        if not rows:
            return 0

        placeholders = ", ".join(["%s"] * len(columns))
        col_str = ", ".join(f'"{c}"' for c in columns)
        query = f'INSERT INTO "{table}" ({col_str}) VALUES ({placeholders})'
        params_list = [tuple(row[c] for c in columns) for row in rows]

        total = await self.execute_many(query, params_list)
        logger.info("Imported %d rows into %s from %s", total, table, filepath)
        return total

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    async def run_migrations(self, migrations_dir: str) -> list[str]:
        """Run numbered SQL migration files from *migrations_dir*.

        Files should be named ``NNN_description.sql`` where NNN is a zero-padded
        integer.  A table ``_schema_migrations`` is created automatically to
        track which migrations have been applied.

        Returns:
            List of migration filenames that were applied.
        """
        await self.execute(
            "CREATE TABLE IF NOT EXISTS _schema_migrations ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  filename TEXT NOT NULL UNIQUE,"
            "  applied_at TEXT NOT NULL"
            ")"
        )

        applied_rows = await self.execute(
            "SELECT filename FROM _schema_migrations ORDER BY id"
        )
        applied: set[str] = {r["filename"] for r in applied_rows}

        mig_path = Path(migrations_dir)
        if not mig_path.is_dir():
            raise FileNotFoundError(f"Migrations directory not found: {migrations_dir}")

        sql_files = sorted(mig_path.glob("*.sql"))
        now = datetime.now(timezone.utc).isoformat()
        executed: list[str] = []

        for sql_file in sql_files:
            fname = sql_file.name
            if fname in applied:
                continue

            sql = sql_file.read_text(encoding="utf-8")
            await self.execute(sql)
            await self.execute(
                "INSERT INTO _schema_migrations (filename, applied_at) VALUES (%s, %s)",
                (fname, now),
            )
            executed.append(fname)
            logger.info("Applied migration: %s", fname)

        return executed

    # ------------------------------------------------------------------
    # Backup (SQLite only)
    # ------------------------------------------------------------------

    async def backup(self, filepath: str) -> None:
        """Create a backup of the database to *filepath*.

        For SQLite the file is copied directly.  For PostgreSQL a
        ``pg_dump`` is attempted via subprocess.
        """
        if self.db_type == "sqlite":
            if not self._sqlite_path or self._sqlite_path == ":memory:":
                raise RuntimeError("Cannot backup an in-memory SQLite database.")
            parent = os.path.dirname(filepath)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.copy2(self._sqlite_path, filepath)
            logger.info("SQLite backup created at %s", filepath)
        else:
            import asyncio

            import shlex

            parts = shlex.split(self.connection_string)
            # postgresql://user:pass@host:port/dbname
            uri = parts[0] if parts else self.connection_string
            pattern = r"postgresql://([^:]+)(?::([^@]*))?@([^:]+)(?::(\d+))?/(.+)"
            match = re.match(pattern, uri)
            if not match:
                raise ValueError("Cannot parse PostgreSQL connection string for backup.")
            user, password, host, port, dbname = match.groups()
            port = port or "5432"

            env = os.environ.copy()
            if password:
                env["PGPASSWORD"] = password

            cmd = [
                "pg_dump",
                "-h", host,
                "-p", port,
                "-U", user,
                "-F", "c",
                "-f", filepath,
                dbname,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"pg_dump failed: {stderr.decode()}")
            logger.info("PostgreSQL backup created at %s", filepath)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict:
        """Return database statistics: size, table count, row counts, etc."""
        tables = await self.get_tables()
        total_rows = sum(t["row_count"] for t in tables)

        size_bytes = 0
        if self.db_type == "sqlite":
            if self._sqlite_path and self._sqlite_path != ":memory:" and os.path.isfile(self._sqlite_path):
                size_bytes = os.path.getsize(self._sqlite_path)
        else:
            db_name_rows = await self.execute("SELECT current_database() AS db")
            if db_name_rows:
                db_name = db_name_rows[0]["db"]
                size_rows = await self.execute(
                    "SELECT pg_database_size(%s) AS size", (db_name,)
                )
                if size_rows:
                    size_bytes = size_rows[0]["size"]

        return {
            "db_type": self.db_type,
            "size_bytes": size_bytes,
            "size_human": _human_bytes(size_bytes),
            "table_count": len(tables),
            "total_rows": total_rows,
            "tables": tables,
        }

    # ------------------------------------------------------------------
    # Transaction support
    # ------------------------------------------------------------------

    async def begin_tx(self) -> None:
        """Begin an explicit transaction."""
        if self.db_type == "sqlite":
            self._cursor.execute("BEGIN")
        else:
            self._connection.autocommit = False
            self._cursor.execute("BEGIN")
        self._in_transaction = True
        logger.debug("Transaction started.")

    async def commit(self) -> None:
        """Commit the current transaction."""
        if not self._in_transaction:
            return
        if self.db_type == "sqlite":
            self._connection.commit()
        else:
            self._connection.commit()
            self._connection.autocommit = True
        self._in_transaction = False
        logger.debug("Transaction committed.")

    async def rollback(self) -> None:
        """Roll back the current transaction."""
        if not self._in_transaction:
            return
        if self.db_type == "sqlite":
            self._connection.rollback()
        else:
            self._connection.rollback()
            self._connection.autocommit = True
        self._in_transaction = False
        logger.debug("Transaction rolled back.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_connected(self) -> None:
        """Raise if no active connection, otherwise reconnect if dropped."""
        if self._connection is None:
            await self.connect()
        try:
            if self.db_type == "sqlite":
                self._connection.execute("SELECT 1")
            else:
                self._cursor.execute("SELECT 1")
        except Exception:
            await self.connect()


def _human_bytes(num_bytes: int) -> str:
    """Convert a byte count to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"
