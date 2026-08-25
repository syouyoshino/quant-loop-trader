"""Autonomous research loop — L4.

OBSERVATION MODE by default: runs experiments + full validation gate, stores
knowledge, but never promotes champions or alters scientific standards without
explicit human approval (--approve-champions is intentionally NOT implemented;
promotion stays a human decision per REVIEW MODE).

Compute governance: hard experiment budget + duplicate prevention via the
experiments ledger. Scheduling: invoke this module from cron/launchd — one
invocation = one bounded research session (crash-safe: partial sessions leave
valid stored experiments).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import logging
from datetime import datetime, timezone

import duckdb

from quant_loop_trader.data import DB_PATH, migrate_db
from quant_loop_trader.experiment import run_experiment
from quant_loop_trader.agents import validate_experiment
from quant_loop_trader.research_memory import search_memory, duplicate_risk

logger = logging.getLogger(__name__)

# Candidate research grid — the "research director" priority queue.
# Ordered by expected information gain: unseen windows first, then unseen seeds.
GRID = [
    {"start": a, "end": b, "seed": s}
    for a, b, s in itertools.product(
        ["2018-01-01", "2020-01-01", "2022-01-01"],
        ["2022-12-31", "2023-06-30", "2024-12-31"],
        [42, 123, 777],
    )
]


def _config_key(ticker: str, horizon: int, start: str, end: str, seed: int) -> str:
    return hashlib.sha256(f"{ticker}{horizon}{seed}{start}{end}".encode()).hexdigest()[:8]


def _already_run(ticker: str, horizon: int, start: str, end: str, seed: int) -> bool:
    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    key = f"%{_config_key(ticker, horizon, start, end, seed)}%"
    n = con.execute("SELECT count(*) FROM experiments WHERE experiment_id LIKE ?", [key]).fetchone()[0]
    con.close()
    return n > 0


def _frontier_remaining(ticker: str = "SPY", horizon: int = 5) -> int:
    return sum(1 for c in GRID if not _already_run(ticker, horizon, c["start"], c["end"], c["seed"]))


def review_memory() -> dict:
    """Research director step 1: what do we already believe?"""
    fails = search_memory("volatility regime")
    return {
        "total_memory_rows": len(fails),
        "recent_beliefs": [{"type": r["memory_type"], "confidence": r["confidence"]} for r in fails[:3]],
    }


def select_candidates(ticker: str, horizon: int, budget: int) -> list[dict]:
    """Priority queue: skip duplicates, cap at remaining budget."""
    out = []
    for cand in GRID:
        if len(out) >= budget:
            break
        if _already_run(ticker, horizon, cand["start"], cand["end"], cand["seed"]):
            continue
        out.append(cand)
    return out


def run_session(ticker: str = "SPY", horizon: int = 5,
                max_experiments: int = 3, validate: bool = True) -> dict:
    """One bounded autonomous research session.

    SAFETY (audit AD3): gated on env QLT_AUTONOMOUS_ENABLED=true — the launchd path
    was previously ungated. Lockfile-guarded; heartbeat written even on crash."""
    if os.getenv("QLT_AUTONOMOUS_ENABLED", "").lower() != "true":
        logger.warning(json.dumps({"event": "session_blocked", "reason": "QLT_AUTONOMOUS_ENABLED not true"}))
        return {"mode": "OBSERVATION", "executed": 0, "results": [], "skipped": "autonomous_disabled"}

    import fcntl
    ROOT = DB_PATH.parent
    lock_path = ROOT / ".session.lock"
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.warning(json.dumps({"event": "session_locked", "detail": "another session holds the lock"}))
        return {"mode": "OBSERVATION", "executed": 0, "results": [], "skipped": "session_locked"}

    try:
        summary = _run_session_body(ticker, horizon, max_experiments, validate, ROOT)
    except Exception as e:
        # a crashed session must be LOUD (audit H5/AD4): alert + crashed heartbeat
        from quant_loop_trader.monitoring.heartbeat import write_heartbeat
        from quant_loop_trader.monitoring.alerts import send_alert
        write_heartbeat(ROOT / "logs", status="crashed", details={"error": str(e)[:200]})
        send_alert("session_crashed", "critical", {"error": str(e)[:200]})
        raise
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
    return summary


def _run_session_body(ticker: str, horizon: int, max_experiments: int,
                      validate: bool, ROOT) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    memory_review = review_memory()
    candidates = select_candidates(ticker, horizon, max_experiments)

    results = []
    for i, cand in enumerate(candidates):
        logger.info(json.dumps({"event": "session_experiment_start", "index": i + 1,
                                "config": cand, "remaining_budget": max_experiments - i}))
        report = run_experiment(ticker=ticker, horizon=horizon,
                                start=cand["start"], end=cand["end"], seed=cand["seed"])
        entry = {
            "experiment_id": report["experiment_id"],
            "decision": report["decision"],
            "delta_acc": report["improvement_delta_accuracy"],
            "validation_status": None,
            "issues": None,
        }
        if validate:
            verdict = validate_experiment(report["experiment_id"])
            entry["validation_status"] = verdict["approval_status"]
            entry["issues"] = verdict["issues_found"]
        results.append(entry)

    # heartbeat + alerting via monitoring layer (distinguishes healthy-idle from broken-silent)
    from quant_loop_trader.monitoring.heartbeat import write_heartbeat
    from quant_loop_trader.monitoring.alerts import send_alert
    remaining = _frontier_remaining(ticker, horizon)
    write_heartbeat(ROOT / "logs", status="healthy",
                    last_task=results[-1]["experiment_id"] if results else None,
                    details={"executed": len(results),
                             "decisions": [r["decision"] for r in results],
                             "grid_remaining": remaining})
    if len(results) == 0:
        send_alert("research_grid_exhausted", "warning",
                   {"ticker": ticker, "horizon": horizon,
                    "note": "no new candidates — hypothesis refresh required"})
    # nightly insurance: checkpointed DB backup, keep last 8 (torn-copy guard, audit L4)
    bdir = ROOT / "backups"
    bdir.mkdir(exist_ok=True)
    import shutil
    con = duckdb.connect(str(DB_PATH))
    con.execute("CHECKPOINT")
    con.close()
    shutil.copy2(DB_PATH, bdir / f"research_{datetime.now(timezone.utc).strftime('%Y%m%d')}.duckdb")
    for old in sorted(bdir.glob("research_*.duckdb"))[:-8]:
        old.unlink()

    summary = {
        "session_started": started,
        "session_finished": datetime.now(timezone.utc).isoformat(),
        "mode": "OBSERVATION",
        "memory_review": memory_review,  # research-director traceability: what we believed entering the session
        "budget": max_experiments,
        "executed": len(results),
        "grid_remaining": remaining,
        "results": results,
    }
    logger.info(json.dumps({"event": "session_complete", "executed": len(results)}))
    return summary


def main():
    p = argparse.ArgumentParser(description="Autonomous research session (observation mode)")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--max-experiments", type=int, default=3)
    p.add_argument("--no-validate", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = run_session(args.ticker, args.horizon, args.max_experiments, not args.no_validate)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
