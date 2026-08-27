"""Read-only access to Quant Loop state.

Every function here READS. Nothing in this module opens a writable DuckDB
connection, writes a file, or calls ``migrate_db`` (which would write). The
dashboard is a window into the research system, never a source of truth.

Deliberately imports neither ``polars`` nor ``quant_loop_trader.data``: the
dashboard must stay a light process that cannot touch research state.
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

# Poll-friendly memoisation: one dashboard refresh hits the same artifacts from
# several panels. Set to 0 to disable (tests do).
CACHE_TTL = float(os.environ.get("QLT_DASHBOARD_CACHE_TTL", "5"))
_CACHES: list[dict] = []


def ttl_cache(fn):
    """Cache by args for CACHE_TTL seconds. Read-only data only."""
    store: dict = {}
    _CACHES.append(store)

    @functools.wraps(fn)
    def inner(*args):
        if CACHE_TTL <= 0:
            return fn(*args)
        now = time.monotonic()
        hit = store.get(args)
        if hit is not None and now - hit[0] < CACHE_TTL:
            return hit[1]
        value = fn(*args)
        store[args] = (now, value)
        return value

    return inner


def clear_caches() -> None:
    for store in _CACHES:
        store.clear()

_PKG_ROOT = Path(__file__).resolve().parents[3]  # …/quant-loop-trader (src/pkg/dashboard/…)


def root() -> Path:
    env = os.environ.get("QLT_ROOT")
    if env:
        return Path(env)
    if (_PKG_ROOT / "data").exists():
        return _PKG_ROOT
    return Path.cwd()


def paths() -> dict:
    r = root()
    return {
        "root": r,
        "data": r / "data",
        "db": r / "data" / "research.duckdb",
        "experiments": r / "data" / "experiments",
        "datasets": r / "data" / "datasets",
        "processed": r / "data" / "processed",
        "logs": r / "data" / "logs",
        "reports": r / "data" / "reports",
    }


class DataUnavailable(RuntimeError):
    """Raised when a source cannot be read. Callers surface NOT AVAILABLE."""


# --- DuckDB (read-only, always) --------------------------------------------
def connect() -> duckdb.DuckDBPyConnection:
    """Open the research database read-only. Never migrates, never writes."""
    db = paths()["db"]
    if not db.exists():
        raise DataUnavailable(f"database_missing:{db}")
    try:
        return duckdb.connect(str(db), read_only=True)
    except duckdb.IOException:
        time.sleep(0.5)  # a writer session holds the lock — contention, not corruption
        try:
            return duckdb.connect(str(db), read_only=True)
        except duckdb.IOException as exc:
            raise DataUnavailable(f"database_locked:{str(exc)[:80]}") from exc


def query(sql: str, params: list | None = None) -> list[dict]:
    con = connect()
    try:
        cur = con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        con.close()


def db_status() -> dict:
    try:
        n = query("SELECT count(*) AS n FROM experiments")[0]["n"]
        return {"status": "OK", "experiment_rows": n}
    except DataUnavailable as exc:
        return {"status": "UNAVAILABLE", "detail": str(exc)}
    except Exception as exc:  # corrupt / unmigrated database
        return {"status": "ERROR", "detail": str(exc)[:120]}


@ttl_cache
def experiment_rows() -> list[dict]:
    return query(
        "SELECT experiment_id, dataset_id, ticker, horizon_days, hypothesis, "
        "economic_reasoning, research_question, model_version, feature_version, seed, "
        "config_json, metrics_json, decision, parent_experiment_id, created_at, "
        "authoritative FROM experiments ORDER BY created_at"
    )


@ttl_cache
def model_registry_rows() -> list[dict]:
    return query(
        "SELECT model_id, parent_model_id, training_data_version, feature_version, "
        "status, research_lineage, performance_history_json, failure_modes, created_at "
        "FROM model_registry ORDER BY created_at"
    )


@ttl_cache
def research_memory_rows() -> list[dict]:
    return query(
        "SELECT memory_id, experiment_id, memory_type, hypothesis, outcome, lesson, "
        "confidence, created_at, authoritative FROM research_memory "
        "WHERE authoritative ORDER BY created_at DESC"
    )


@ttl_cache
def dataset_rows() -> list[dict]:
    return query(
        "SELECT dataset_id, ticker, start_date, end_date, source, row_count, "
        "validation_status, checksum, created_at FROM datasets ORDER BY created_at"
    )


def task_counts() -> dict:
    rows = query("SELECT status, count(*) AS n FROM tasks GROUP BY status")
    return {r["status"]: r["n"] for r in rows}


def parquet_rows(path: Path, columns: str = "*", order: str = "") -> list[dict]:
    """Read a parquet file through DuckDB (keeps polars out of the dashboard)."""
    if not Path(path).exists():
        raise DataUnavailable(f"parquet_missing:{path}")
    con = duckdb.connect()  # in-memory scratch; touches nothing on disk
    try:
        sql = f"SELECT {columns} FROM read_parquet(?)" + (f" ORDER BY {order}" if order else "")
        cur = con.execute(sql, [str(path)])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        con.close()


# --- experiment artifacts ---------------------------------------------------
ARTIFACTS = ("report.json", "metrics.json", "config.json", "validation.json",
             "holdout_report.json", "predictions.lock")


@ttl_cache
def experiment_ids() -> list[str]:
    exp_root = paths()["experiments"]
    if not exp_root.exists():
        return []
    return sorted(p.name for p in exp_root.iterdir() if p.is_dir())


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, NotADirectoryError):
        return None
    except json.JSONDecodeError:
        return None


@ttl_cache
def artifacts(experiment_id: str) -> dict:
    """All sealed artifacts of one experiment, plus filesystem timing evidence."""
    d = paths()["experiments"] / experiment_id
    out = {"experiment_id": experiment_id, "dir": d, "exists": d.is_dir()}
    for name in ARTIFACTS:
        out[name.split(".")[0]] = read_json(d / name)
    out["files"] = {name: (d / name).exists() for name in ARTIFACTS}
    out["started_at"] = _birthtime(d)
    out["sealed_at"] = _mtime(d / "report.json")
    out["validated_at"] = _mtime(d / "validation.json")
    out["holdout_at"] = _mtime(d / "holdout_report.json")
    return out


def _stat(path: Path):
    try:
        return Path(path).stat()
    except OSError:
        return None


def _mtime(path: Path) -> str | None:
    st = _stat(path)
    return datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat() if st else None


def _birthtime(path: Path) -> str | None:
    st = _stat(path)
    if not st:
        return None
    ts = getattr(st, "st_birthtime", st.st_mtime)
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def predictions(experiment_id: str, variant: str = "improved") -> list[dict]:
    p = paths()["experiments"] / experiment_id / f"predictions_{variant}.parquet"
    return parquet_rows(p, order="event_time")


@ttl_cache
def price_history(ticker: str = "SPY") -> list[dict]:
    return parquet_rows(paths()["processed"] / f"{ticker}.parquet", order="event_time")


# --- autonomy / system state ------------------------------------------------
@ttl_cache
def heartbeat() -> dict | None:
    return read_json(paths()["logs"] / "heartbeat.json")


@ttl_cache
def session_records() -> list[dict]:
    """Autonomy session summaries appended to session.log as pretty JSON blobs."""
    log = paths()["logs"] / "session.log"
    if not log.exists():
        return []
    text = log.read_text()
    decoder = json.JSONDecoder()
    out, i = [], 0
    while i < len(text):
        start = text.find("{", i)
        if start < 0:
            break
        try:
            obj, i = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        if isinstance(obj, dict) and "session_started" in obj:
            out.append(obj)
    out.sort(key=lambda r: r["session_started"])
    return out


def log_tail(name: str, limit: int = 4000) -> str:
    p = paths()["logs"] / name
    if not p.exists():
        return ""
    return p.read_text()[-limit:]


def autonomy_enabled() -> bool | None:
    """QLT_AUTONOMOUS_ENABLED gates every session; also set by the launchd plists."""
    env = os.environ.get("QLT_AUTONOMOUS_ENABLED")
    if env is not None:
        return env.lower() == "true"
    hits = []
    for plist in sorted((root() / "deploy").glob("*.plist")):
        text = plist.read_text()
        if "QLT_AUTONOMOUS_ENABLED" in text:
            after = text.split("QLT_AUTONOMOUS_ENABLED", 1)[1]
            hits.append("<string>true</string>" in after.split("</dict>")[0])
    if hits:
        return any(hits)
    return None


@ttl_cache
def scheduled_jobs() -> list[dict]:
    """launchd jobs declared by deploy/*.plist and whether launchctl knows them."""
    jobs = []
    for plist in sorted((root() / "deploy").glob("*.plist")):
        text = plist.read_text()
        label = text.split("<key>Label</key>", 1)[-1].split("<string>", 1)[-1].split("</string>")[0].strip()
        jobs.append({"label": label, "plist": plist.name, "loaded": _launchctl_loaded(label)})
    return jobs


def _launchctl_loaded(label: str) -> bool | None:
    try:
        r = subprocess.run(["launchctl", "list", label], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return None


@ttl_cache
def git_commit() -> dict:
    try:
        r = subprocess.run(["git", "-C", str(root()), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        commit = r.stdout.strip() or None
        d = subprocess.run(["git", "-C", str(root()), "status", "--porcelain"],
                           capture_output=True, text=True, timeout=5)
        return {"commit": commit, "dirty": bool(d.stdout.strip()) if commit else None}
    except Exception:
        return {"commit": None, "dirty": None}
