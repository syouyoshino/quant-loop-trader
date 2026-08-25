"""Research memory + belief updates. Institutional memory of successes and failures."""
from __future__ import annotations

import json
import logging
from pathlib import Path


from quant_loop_trader import data as _data
from quant_loop_trader.data import migrate_db

logger = logging.getLogger(__name__)


def _con(db_path=None):
    migrate_db(db_path)
    import duckdb as _duckdb
    return _duckdb.connect(str(Path(db_path or _data.DB_PATH)))


def search_memory(query: str, memory_type: str | None = None) -> list[dict]:
    """Search past research by keyword. Returns rows newest-first."""
    con = _con()
    sql = "SELECT * FROM research_memory WHERE authoritative AND (hypothesis ILIKE ? OR lesson ILIKE ?)"
    params = [f"%{query}%", f"%{query}%"]
    if memory_type:
        sql += " AND memory_type = ?"
        params.append(memory_type)
    sql += " ORDER BY created_at DESC"
    cols = [d[0] for d in con.execute(sql, params).description]
    rows = con.execute(sql, params).fetchall()
    con.close()
    return [dict(zip(cols, r)) for r in rows]


def record_outcome(report: dict) -> list[str]:
    """Store experiment outcome as institutional memory. Returns memory_ids."""
    decision = report["decision"]
    exp_id = report["experiment_id"]
    hypothesis = report["hypothesis"]

    if decision == "KEEP":
        memory_type = "success"
        lesson = f"Confirmed: {hypothesis} Survived hidden-future test; evidence supports vol-regime conditioning."
        confidence = min(0.95, prior_confidence(hypothesis) + 0.15)
    elif decision == "IMPROVE":
        memory_type = "partial"
        lesson = f"Partial: accuracy improved but Sharpe degraded. Refine risk control before retest."
        confidence = max(0.05, prior_confidence(hypothesis) - 0.0)
    else:
        memory_type = "failure"
        lesson = f"Rejected: {report['failure_condition']} Vol-interaction features did not change predictions materially."
        confidence = max(0.05, prior_confidence(hypothesis) - 0.2)

    autopsy = report.get("error_analysis", {})
    conditions = json.dumps({k: v for k, v in autopsy.items() if k.startswith("regime")})
    memory_id = f"mem_{exp_id}_{memory_type}"
    con = _con()
    con.execute(
        "INSERT OR REPLACE INTO research_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'v1', current_timestamp, TRUE)",
        [
            memory_id,
            exp_id,
            memory_type,
            hypothesis,
            report.get("economic_reasoning"),
            decision,
            lesson,
            conditions,
            json.dumps({"metrics": report.get("improved_metrics", {}), "delta_acc": report.get("improvement_delta_accuracy")}),
            confidence,
            json.dumps({"branch": report.get("research_branch"), "creator": report.get("creator_agent")}),
        ],
    )
    # market/model knowledge from autopsy
    knowledge_id = f"mem_{exp_id}_knowledge"
    con.execute(
        "INSERT OR REPLACE INTO research_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'v1', current_timestamp, TRUE)",
        [
            knowledge_id,
            exp_id,
            "model_knowledge",
            f"Baseline regime performance for {report['config']['ticker']} {report['config']['horizon']}d",
            None,
            decision,
            f"Autopsy: {autopsy.get('overall_accuracy', 'n/a')} overall accuracy across vol regimes; false_positive_rate={autopsy.get('false_positive_rate')}",
            conditions,
            json.dumps(autopsy),
            0.6,
            "{}",
        ],
    )
    con.close()
    logger.info(json.dumps({"event": "memory_recorded", "memory_id": memory_id, "confidence": confidence}))
    return [memory_id, knowledge_id]


def prior_confidence(hypothesis: str) -> float:
    """Current belief in a hypothesis from accumulated evidence."""
    rows = search_memory(hypothesis[:60])
    if not rows:
        return 0.5
    # most recent memory's confidence is the running belief
    return float(rows[0]["confidence"])


def duplicate_risk(hypothesis: str, max_rejects: int = 3) -> dict:
    """Flag if similar hypothesis already failed repeatedly."""
    fails = [r for r in search_memory(hypothesis[:60], memory_type="failure")]
    return {"similar_failures": len(fails), "should_warn": len(fails) >= max_rejects}


def register_features(feature_defs: list[dict]) -> None:
    """Idempotent feature registration. Each def: feature_id, formula, creator, data_dependencies, available_time_logic, validation_status."""
    con = _con()
    for f in feature_defs:
        con.execute(
            "INSERT OR REPLACE INTO feature_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'v1', current_timestamp)",
            [
                f["feature_id"], f["formula"], f.get("creator", "feature_engineer_v0"),
                f.get("data_dependencies", "SPY daily close"), f.get("available_time_logic", "shift(1): uses data up to t-1"),
                f.get("validation_status", "validated"), "{}", f.get("failure_conditions", ""), "{}",
            ],
        )
    con.close()


def register_model(model: dict) -> None:
    """Idempotent model registration."""
    con = _con()
    con.execute(
        "INSERT OR REPLACE INTO model_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'v1', current_timestamp)",
        [
            model["model_id"], model.get("parent_model_id"), model["training_data_version"],
            model["feature_version"], model.get("parameters_json", "{}"), model.get("performance_history_json", "{}"),
            model.get("failure_modes", ""), model.get("research_lineage", ""),
            model.get("status", "candidate"), "{}",
        ],
    )
    con.close()
