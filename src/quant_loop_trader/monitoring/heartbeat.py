"""Heartbeat (Task 5): the liveness signal for unattended operation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

STALE_AFTER_SECONDS = 48 * 3600  # healthy-idle vs broken-silent boundary


def write_heartbeat(logs_dir: Path, status: str = "healthy", last_task: str | None = None,
                    details: dict | None = None) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "last_task": last_task,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(details or {}),
    }
    out = logs_dir / "heartbeat.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def read_heartbeat(logs_dir: Path) -> dict | None:
    f = Path(logs_dir) / "heartbeat.json"
    return json.loads(f.read_text()) if f.exists() else None


def is_stale(logs_dir: Path, now: datetime | None = None) -> bool:
    hb = read_heartbeat(logs_dir)
    if hb is None:
        return True  # never beat = stale from birth
    ts = datetime.fromisoformat(hb["timestamp"])
    now = now or datetime.now(timezone.utc)
    return (now - ts).total_seconds() > STALE_AFTER_SECONDS
