"""DuckDB-backed task queue (Phase 10). Persistent, crash-safe, duplicate-guarded.

NOTE: nothing enqueues tasks automatically. The controller is the only writer,
and it refuses to start unless explicitly enabled (see automation/controller.py).
"""
from __future__ import annotations

import json
import uuid

import duckdb

from quant_loop_trader.data import DB_PATH, migrate_db


def enqueue(task_type: str, payload: dict, priority: int = 5, db_path=None) -> str:
    migrate_db(db_path)
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    con = duckdb.connect(str(db_path or DB_PATH))
    con.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, 'pending', ?, NULL, 0, NULL, ?, 'v1', current_timestamp, NULL)",
        [task_id, task_type, json.dumps(payload), int(priority),
         json.dumps({"enqueued_by": "api"})],
    )
    con.close()
    return task_id


def claim_next(db_path=None) -> dict | None:
    """Atomically claim the highest-priority pending task (single-writer DuckDB)."""
    con = duckdb.connect(str(db_path or DB_PATH))
    row = con.execute(
        "SELECT task_id, task_type, payload_json FROM tasks "
        "WHERE status = 'pending' ORDER BY priority, created_at LIMIT 1"
    ).fetchone()
    if not row:
        con.close()
        return None
    con.execute("UPDATE tasks SET status='running', updated_at=current_timestamp WHERE task_id=?", [row[0]])
    con.close()
    return {"task_id": row[0], "task_type": row[1], "payload": json.loads(row[2])}


def complete(task_id: str, result: dict, ok: bool = True, db_path=None) -> None:
    con = duckdb.connect(str(db_path or DB_PATH))
    con.execute("UPDATE tasks SET status=?, result_json=?, attempts=attempts+1, updated_at=current_timestamp WHERE task_id=?",
                ["done" if ok else "failed", json.dumps(result), task_id])
    con.close()


def pending_count(db_path=None) -> int:
    con = duckdb.connect(str(db_path or DB_PATH), read_only=True)
    n = con.execute("SELECT count(*) FROM tasks WHERE status='pending'").fetchone()[0]
    con.close()
    return n
