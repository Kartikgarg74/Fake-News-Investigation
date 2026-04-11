"""TrajectoriesDB — RL training artifacts + chain-of-custody audit.

Two tables:
- trajectories: every (episode, step, state, action, reward) tuple for RL training
- audit: every evidence retrieval (source, timestamp, content hash) for auditability

The trajectories table makes this environment RL-trainable: dump it to JSONL,
feed to PPO. The audit table makes it production-credible: "here's exactly
which Wikipedia revision the agent used."
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

from .base import DatabaseManager


class TrajectoriesDB(DatabaseManager):
    filename = "trajectories.db"

    schema = """
    CREATE TABLE IF NOT EXISTS trajectories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        episode_id TEXT NOT NULL,
        step_index INTEGER NOT NULL,
        claim_id TEXT DEFAULT '',
        difficulty TEXT DEFAULT '',
        state_json TEXT DEFAULT '{}',
        action_json TEXT DEFAULT '{}',
        reward REAL DEFAULT 0.0,
        done INTEGER DEFAULT 0,
        created_at INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_traj_episode ON trajectories(episode_id);
    CREATE INDEX IF NOT EXISTS idx_traj_claim ON trajectories(claim_id);

    CREATE TABLE IF NOT EXISTS audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        episode_id TEXT NOT NULL,
        claim_id TEXT DEFAULT '',
        source_url TEXT NOT NULL,
        source_type TEXT DEFAULT '',
        content_hash TEXT NOT NULL,
        fetched_at INTEGER NOT NULL,
        status TEXT DEFAULT 'ok'
    );

    CREATE INDEX IF NOT EXISTS idx_audit_episode ON audit(episode_id);
    CREATE INDEX IF NOT EXISTS idx_audit_source ON audit(source_url);
    """

    def log_step(
        self,
        episode_id: str,
        step_index: int,
        claim_id: str,
        difficulty: str,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        done: bool,
    ) -> bool:
        return self.write(
            """INSERT INTO trajectories
               (episode_id, step_index, claim_id, difficulty,
                state_json, action_json, reward, done, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode_id,
                step_index,
                claim_id,
                difficulty,
                json.dumps(state),
                json.dumps(action),
                float(reward),
                1 if done else 0,
                int(time.time()),
            ),
        )

    def log_audit(
        self,
        episode_id: str,
        claim_id: str,
        source_url: str,
        source_type: str,
        content_hash: str,
        status: str = "ok",
    ) -> bool:
        return self.write(
            """INSERT INTO audit
               (episode_id, claim_id, source_url, source_type,
                content_hash, fetched_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                episode_id, claim_id, source_url, source_type,
                content_hash, int(time.time()), status,
            ),
        )

    def get_episode(self, episode_id: str) -> List[Dict[str, Any]]:
        rows = self.execute(
            "SELECT * FROM trajectories WHERE episode_id = ? ORDER BY step_index",
            (episode_id,),
        )
        return [dict(r) for r in rows]

    def export_jsonl(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Export all trajectories as a list of dicts, ready for JSONL dump."""
        sql = "SELECT * FROM trajectories ORDER BY episode_id, step_index"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self.execute(sql)
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["state"] = json.loads(d.get("state_json") or "{}")
            except Exception:
                d["state"] = {}
            try:
                d["action"] = json.loads(d.get("action_json") or "{}")
            except Exception:
                d["action"] = {}
            d.pop("state_json", None)
            d.pop("action_json", None)
            out.append(d)
        return out

    def count_steps(self) -> int:
        row = self.execute_one("SELECT COUNT(*) FROM trajectories")
        return int(row[0]) if row else 0

    def count_episodes(self) -> int:
        row = self.execute_one("SELECT COUNT(DISTINCT episode_id) FROM trajectories")
        return int(row[0]) if row else 0
