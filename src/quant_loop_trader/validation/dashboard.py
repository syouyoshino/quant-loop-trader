"""Bias dashboard (Phase 7): aggregate the system's own weakness signals.

Reads only stored research state — answers "how likely is our record full of noise?"
without running anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb

from quant_loop_trader import data as _data
from quant_loop_trader.data import migrate_db


def build_dashboard(exp_root: Path, db_path=None) -> dict:
    migrate_db(db_path)
    try:
        con = duckdb.connect(str(db_path or _data.DB_PATH), read_only=True)
    except duckdb.IOException:
        import time
        time.sleep(1)  # writer lock held — contention, not corruption
        con = duckdb.connect(str(db_path or _data.DB_PATH), read_only=True)
    total_exp = con.execute("SELECT count(*) FROM experiments WHERE authoritative AND experiment_id NOT LIKE '%_baseline'").fetchone()[0]
    decisions = dict(con.execute(
        "SELECT decision, count(*) FROM experiments WHERE authoritative AND experiment_id NOT LIKE '%_baseline' GROUP BY decision"
    ).fetchall())
    distinct_hyp = con.execute("SELECT count(DISTINCT hypothesis) FROM experiments WHERE authoritative").fetchone()[0]
    models = dict(con.execute("SELECT status, count(*) FROM model_registry GROUP BY status").fetchall())
    con.close()

    # scan validation artifacts for hostile-gate signal frequencies
    gate_reasons: dict[str, int] = {}
    validations = 0
    unreadable = 0
    for vfile in Path(exp_root).glob("*/validation.json"):
        try:
            v = json.loads(vfile.read_text())
        except Exception:
            unreadable += 1  # silently dropping unreadable evidence would hide compromise
            continue
        # normalize issue keys: strip embedded values so counts don't fragment
        for issue in v.get("issues_found", []):
            key = issue.split(":")[0]
            gate_reasons[key] = gate_reasons.get(key, 0) + 1
        validations += 1

    keep = decisions.get("KEEP", 0)  # single vocabulary: KEEP decisions only
    dashboard = {
        "experiments_total": total_exp,
        "decisions": decisions,
        "keep_ratio": keep / total_exp if total_exp else 0.0,
        # data-mining bias proxy: many experiments, few hypotheses → re-testing one idea
        "hypotheses_tested": distinct_hyp,
        "experiments_per_hypothesis": round(total_exp / distinct_hyp, 2) if distinct_hyp else None,
        # selection-bias proxy: a KEEP ratio far above base rate with low validation pass rate
        "validations_run": validations,
        "gate_rejection_reasons": dict(sorted(gate_reasons.items(), key=lambda kv: -kv[1])),
        "model_registry": models,
        # leakage-risk sentinel: degenerate predictors must be zero among candidates
        "degenerate_flags": gate_reasons.get("degenerate_constant_predictions", 0)
                            + gate_reasons.get("feature_shuffle", 0),
        "unreadable_validation_files": unreadable,
        "verdict": _verdict(total_exp, keep, distinct_hyp, gate_reasons),
    }
    return dashboard


def _verdict(total: int, keeps: int, hyps: int, reasons: dict) -> str:
    if total == 0:
        return "NO_RESEARCH_YET"
    if reasons.get("artifact_tampered", 0) > 0:
        return "COMPROMISED"
    if total >= 10 and keeps / max(hyps, 1) > 0.5:
        return "SUSPICIOUS_KEEP_RATE"  # too many wins = mining
    return "HEALTHY_SKEPTICAL" if keeps <= max(1, total // 10) else "REVIEW"
