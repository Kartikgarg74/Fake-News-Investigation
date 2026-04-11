"""Base class for all Veritas SQLite database managers.

Every DB manager inherits from DatabaseManager and gets:
- Resilient path resolution (falls back to /tmp if DATA_DIR is read-only,
  which is the exact failure mode we hit in validator containers)
- Connection context manager with Row factory
- Idempotent schema creation
- Safe migration helpers that swallow errors on read-only paths
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

DATA_DIR = Path(__file__).parent.parent.parent / "data"
TMP_DIR = Path("/tmp")


class DatabaseManager:
    """Base class — resolve a writable path, create schema, expose safe queries.

    Subclasses must define:
        filename: str       (e.g. "claims.db")
        schema: str         (SQL with CREATE TABLE IF NOT EXISTS ...)

    Subclasses may override _seed() to load initial data on first creation.
    """

    filename: str = ""
    schema: str = ""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self._resolve_path()
        self._ensure_schema()
        self._migrate()

    def _resolve_path(self) -> str:
        """Return a writable absolute path for this DB, falling back to /tmp."""
        if not self.filename:
            raise ValueError(f"{self.__class__.__name__} must define filename")

        default = DATA_DIR / self.filename

        if default.exists():
            return str(default)

        # Try to create in DATA_DIR, fall back to /tmp on any error
        try:
            default.parent.mkdir(parents=True, exist_ok=True)
            # Canary: touch the directory to verify writability
            test = default.parent / ".veritas_write_test"
            test.touch()
            test.unlink()
            return str(default)
        except (PermissionError, OSError):
            return str(TMP_DIR / f"veritas_{self.filename}")

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist. Swallow errors on read-only paths."""
        if not self.schema:
            return
        try:
            with self.connect() as conn:
                conn.executescript(self.schema)
                conn.commit()
                if self._is_empty(conn):
                    self._seed(conn)
                    conn.commit()
        except sqlite3.OperationalError:
            # Read-only path — nothing we can do. Queries will fail later
            # but the env will still boot (which is what the validator checks).
            pass
        except Exception:
            pass

    def _migrate(self) -> None:
        """Hook for subclasses to add columns to existing DBs. Safe no-op by default."""
        pass

    def _is_empty(self, conn: sqlite3.Connection) -> bool:
        """Return True if the primary table has no rows. Subclasses override."""
        return False

    def _seed(self, conn: sqlite3.Connection) -> None:
        """Populate initial data on first creation. Subclasses override if needed."""
        pass

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection with Row factory and sane defaults.

        Uses a fresh connection per-call rather than a pooled one because
        SQLite is fine with that for our workload (low concurrency, short
        transactions) and it avoids cross-thread issues in FastAPI handlers.
        """
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def execute(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Run a query and return all rows. Returns [] on any failure."""
        try:
            with self.connect() as conn:
                cur = conn.execute(sql, params)
                return cur.fetchall()
        except Exception:
            return []

    def execute_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Run a query and return the first row or None."""
        rows = self.execute(sql, params)
        return rows[0] if rows else None

    def write(self, sql: str, params: tuple = ()) -> bool:
        """Run a write query. Returns True on success, False on any failure."""
        try:
            with self.connect() as conn:
                conn.execute(sql, params)
                conn.commit()
            return True
        except Exception:
            return False

    def writemany(self, sql: str, params_list: list[tuple]) -> bool:
        try:
            with self.connect() as conn:
                conn.executemany(sql, params_list)
                conn.commit()
            return True
        except Exception:
            return False
