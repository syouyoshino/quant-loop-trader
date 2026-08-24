"""Health aggregation (Task 5): one call answers 'is the lab alive and sane?'"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from quant_loop_trader.data import DB_PATH
from quant_loop_trader.monitoring.heartbeat import is_stale


def check_health(logs_dir: Path, exp_root: Path, db_path: Path = DB_PATH) -> dict:
    report: dict = {"status": "healthy", "checks": {}}

    # 1. heartbeat liveness
    stale = is_stale(Path(logs_dir))
    report["checks"]["heartbeat"] = "stale" if stale else "ok"
    if stale:
        report["status"] = "degraded"

    # 2. database reachable + experiment activity
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        last_exp = con.execute(
            "SELECT max(created_at) FROM experiments"
        ).fetchone()[0]
        pending = con.execute("SELECT count(*) FROM tasks WHERE status='pending'").fetchone()[0]
        failed = con.execute("SELECT count(*) FROM tasks WHERE status='failed'").fetchone()[0]
        con.close()
        report["checks"]["database"] = "ok"
        report["last_experiment_at"] = str(last_exp)
        report["queue"] = {"pending": pending, "failed": failed}
        if failed > 3:
            report["status"] = max(report["status"], "degraded")
    except Exception as e:
        report["checks"]["database"] = f"error:{str(e)[:80]}"
        report["status"] = "broken"

    # 3. recent experiment artifacts on disk
    recent = sorted(
        (p for p in Path(exp_root).glob("*/report.json")),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    report["last_experiment_artifact"] = recent[0].parent.name if recent else None

    return report


if __name__ == "__main__":
    print(json.dumps(check_health(Path("data/logs"), Path("data/experiments")), indent=2))
