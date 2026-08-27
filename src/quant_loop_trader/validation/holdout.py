"""Permanent out-of-time holdout and final adjudication."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from quant_loop_trader.features import improved_feature_columns

HOLDOUT_FRACTION = 0.15


class HoldoutIntegrityError(RuntimeError):
    """Raised when final holdout evidence is missing, inconsistent, or tampered."""


def holdout_boundary(start: str, end: str) -> str:
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    days = int((e - s).days * HOLDOUT_FRACTION)
    return (e - __import__("datetime").timedelta(days=days)).date().isoformat()


def apply_holdout(df, start: str, end: str, use_holdout: bool):
    import polars as pl

    b = pl.lit(holdout_boundary(start, end)).str.strptime(pl.Date, "%Y-%m-%d")
    if use_holdout:
        return df.filter(pl.col("event_time") >= b)
    return df.filter(pl.col("event_time") < b)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    """Write JSON by atomic rename so readers never observe a partial artifact."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _seal_holdout_evidence(experiment_id: str, bundle, result: dict) -> None:
    """Bind final adjudication to the sealed research bundle and dataset snapshot."""
    exp_dir = bundle.exp_dir
    report_path = exp_dir / "holdout_report.json"
    research_lock_path = exp_dir / "predictions.lock"
    _atomic_json(report_path, result)
    seal = {
        "version": 1,
        "experiment_id": experiment_id,
        "model_id": f"{experiment_id}_improved",
        "holdout_report_sha256": _sha(report_path),
        "research_lock_sha256": _sha(research_lock_path),
        "dataset_snapshot_sha256": _sha(bundle.dataset_snapshot),
        "sealed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(exp_dir / "holdout.lock", seal)


def verify_holdout_evidence(experiment_id: str, exp_root: Path | None = None) -> dict:
    """Return final holdout evidence only when file and DB commitments agree.

    A report written before a crash is not authoritative until the holdout claim
    and registry transition are committed. This closes the filesystem/DB seam.
    """
    import duckdb

    from quant_loop_trader.data import DB_PATH
    from quant_loop_trader.experiment import EXP_ROOT

    root = Path(exp_root) if exp_root is not None else EXP_ROOT
    exp_dir = root / experiment_id
    report_path = exp_dir / "holdout_report.json"
    seal_path = exp_dir / "holdout.lock"
    research_lock_path = exp_dir / "predictions.lock"
    if not report_path.exists() or not seal_path.exists():
        raise HoldoutIntegrityError("missing_holdout_report_or_lock")
    try:
        result = json.loads(report_path.read_text())
        seal = json.loads(seal_path.read_text())
    except json.JSONDecodeError as exc:
        raise HoldoutIntegrityError("invalid_holdout_json") from exc

    issues = []
    if seal.get("experiment_id") != experiment_id:
        issues.append("experiment_binding_mismatch")
    if seal.get("model_id") != f"{experiment_id}_improved":
        issues.append("model_binding_mismatch")
    if seal.get("holdout_report_sha256") != _sha(report_path):
        issues.append("holdout_report_tampered")
    if not research_lock_path.exists() or seal.get("research_lock_sha256") != _sha(research_lock_path):
        issues.append("research_lock_mismatch")

    from quant_loop_trader.bundle import BundleIntegrityError, ExperimentBundle

    try:
        bundle = ExperimentBundle.open_verified(experiment_id, root)
        if seal.get("dataset_snapshot_sha256") != _sha(bundle.dataset_snapshot):
            issues.append("dataset_snapshot_mismatch")
    except BundleIntegrityError as exc:
        issues.append(f"research_bundle_invalid:{str(exc)[:120]}")

    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        claim = con.execute(
            "SELECT state, promoted, result_json FROM holdout_claims WHERE experiment_id=?",
            [experiment_id],
        ).fetchone()
        registry = con.execute(
            "SELECT status FROM model_registry WHERE model_id=?",
            [f"{experiment_id}_improved"],
        ).fetchone()
        con.close()
    except Exception as exc:
        raise HoldoutIntegrityError(f"holdout_db_unavailable:{str(exc)[:100]}") from exc

    if not claim or claim[0] not in {"COMPLETE", "FAILED"}:
        issues.append(f"holdout_claim_not_committed:{claim[0] if claim else 'missing'}")
    else:
        try:
            stored = json.loads(claim[2]) if claim[2] else None
        except json.JSONDecodeError:
            stored = None
        if stored != result:
            issues.append("holdout_db_result_mismatch")
        if bool(claim[1]) != bool(result.get("promoted")):
            issues.append("holdout_db_promotion_mismatch")

    status = registry[0] if registry else None
    if result.get("promoted") and status != "champion":
        issues.append(f"promoted_without_champion_registry:{status or 'missing'}")
    if not result.get("promoted") and status == "champion":
        issues.append("champion_registry_without_promoted_holdout")
    if issues:
        raise HoldoutIntegrityError(";".join(issues))
    return result


def adjudicate_holdout(experiment_id: str) -> dict:
    """Evaluate an eligible model exactly once on its sealed dataset snapshot."""
    import logging

    import duckdb

    from quant_loop_trader.bundle import BundleIntegrityError, ExperimentBundle
    from quant_loop_trader.data import DB_PATH
    from quant_loop_trader.experiment import EXP_ROOT
    from quant_loop_trader.models.registry import build_model

    exp_dir = EXP_ROOT / experiment_id
    try:
        bundle = ExperimentBundle.open_verified(experiment_id, EXP_ROOT)
    except BundleIntegrityError as exc:
        return {"promoted": False, "reason": f"bundle_integrity:{str(exc)[:150]}"}

    cfg = bundle.config
    con = duckdb.connect(str(DB_PATH))
    row = con.execute(
        "SELECT status FROM model_registry WHERE model_id=?",
        [f"{experiment_id}_improved"],
    ).fetchone()
    con.close()
    if not row or row[0] != "eligible":
        return {"promoted": False, "reason": f"not_eligible:{row[0] if row else 'missing'}"}

    claim = _claim_holdout(experiment_id)
    if claim is not None:
        return claim

    h = cfg["horizon"]
    pq = bundle.dataset_snapshot
    fin = {}
    try:
        df = _load_holdout_frame(pq, cfg)
        cols = improved_feature_columns()
        feats = df.select(cols + ["label", "close"])
        Xte = feats.select(cols).to_numpy()
        yte = feats["label"].to_numpy()
        Xtr, ytr = _train_all_nonholdout(pq, cfg)
        model = build_model(cfg.get("model_type", "logistic"), seed=cfg["seed"])
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        base = float(max(yte.mean(), 1 - yte.mean()))
        acc = float((ypred == yte).mean())

        from quant_loop_trader.evaluation import evaluate

        prices_h = feats["close"].to_numpy()
        try:
            prob = model.predict_proba(Xte)
        except Exception:
            prob = ypred.astype(float)
        fin = evaluate(yte, ypred, prob, prices_h, horizon=h)
        economic_gate = (
            fin.get("cumulative_return_strategy_liquidated", fin["cumulative_return_strategy"]) > 0
            and fin.get("sharpe_strategy_liquidated", fin["sharpe_strategy"])
            >= fin["sharpe_benchmark"]
        )

        from quant_loop_trader.core import significance

        sig = significance(yte, np.asarray(ypred), horizon=h)
        sig_ok = sig.passed and sig.n_effective >= 10
        promoted = bool(acc > base and economic_gate and sig_ok)
    except Exception as exc:
        import traceback

        logger = logging.getLogger(__name__)
        logger.error(f"adjudication failed: {exc}")
        result = {
            "promoted": False,
            "reason": f"adjudication_error:{str(exc)[:150]}",
            "economic_gate": fin or None,
            "traceback_tail": traceback.format_exc()[-300:],
        }
        try:
            _seal_holdout_evidence(experiment_id, bundle, result)
            _commit_holdout_outcome(experiment_id, "FAILED", result)
        except Exception as commit_exc:
            logger.error(f"failed to commit holdout failure evidence: {commit_exc}")
        return result

    liq_return = fin.get("cumulative_return_strategy_liquidated", fin["cumulative_return_strategy"])
    liq_sharpe = fin.get("sharpe_strategy_liquidated", fin["sharpe_strategy"])
    result = {
        "promoted": promoted,
        "holdout_accuracy": acc,
        "base_rate": base,
        "n_holdout": int(len(yte)),
        "economic_gate": {
            "compounded_net_return": liq_return,
            "sharpe_strategy": liq_sharpe,
            "sharpe_benchmark": fin["sharpe_benchmark"],
            "liquidated_at_end": True,
        },
        "holdout_metrics": fin,
    }

    from quant_loop_trader.core import LifecycleEvidence, final_state

    new_status = final_state(LifecycleEvidence(
        research_screen=bundle.report["decision"],
        validation="PASS",
        holdout="PASS" if promoted else "FAIL",
    ))
    try:
        _seal_holdout_evidence(experiment_id, bundle, result)
        _commit_holdout_outcome(experiment_id, "COMPLETE", result, new_status=new_status)
    except Exception as exc:
        logging.getLogger(__name__).error(f"holdout finalization failed: {exc}")
        return {
            "promoted": False,
            "reason": f"holdout_commit_error:{str(exc)[:150]}",
        }

    from quant_loop_trader.agents import _correct_success_memories_for

    try:
        if promoted:
            _confirm_success_memory(experiment_id)
        else:
            _correct_success_memories_for(experiment_id)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "holdout committed but memory update failed for %s: %s", experiment_id, exc
        )
    return result


def _claim_holdout(experiment_id: str) -> dict | None:
    """Take the one-shot lock before the holdout is exposed."""
    import duckdb

    from quant_loop_trader.data import DB_PATH, migrate_db

    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    try:
        prior = con.execute(
            "SELECT state FROM holdout_claims WHERE experiment_id=?", [experiment_id]
        ).fetchone()
        if prior:
            return {"promoted": False, "reason": f"holdout_already_consumed:{prior[0]}"}
        con.execute(
            "INSERT INTO holdout_claims (experiment_id, state) VALUES (?, 'CLAIMED')",
            [experiment_id],
        )
    except duckdb.ConstraintException:
        return {"promoted": False, "reason": "holdout_already_consumed:CLAIMED"}
    finally:
        con.close()
    return None


def _commit_holdout_outcome(experiment_id: str, state: str, result: dict,
                            new_status: str | None = None) -> None:
    """Commit registry transition and holdout claim as one DuckDB transaction."""
    import duckdb

    from quant_loop_trader.data import DB_PATH, migrate_db

    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("BEGIN TRANSACTION")
        claim = con.execute(
            "SELECT state FROM holdout_claims WHERE experiment_id=?", [experiment_id]
        ).fetchone()
        if not claim or claim[0] != "CLAIMED":
            raise RuntimeError(f"holdout_claim_not_open:{claim[0] if claim else 'missing'}")
        if new_status is not None:
            con.execute(
                "UPDATE model_registry SET status=? WHERE model_id=?",
                [new_status, f"{experiment_id}_improved"],
            )
        con.execute(
            "UPDATE holdout_claims SET state=?, completed_at=current_timestamp, "
            "promoted=?, result_json=? WHERE experiment_id=?",
            [state, bool(result.get("promoted")), json.dumps(result, sort_keys=True), experiment_id],
        )
        con.execute("COMMIT")
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        con.close()


def release_holdout_claim(experiment_id: str) -> str:
    """Explicit human recovery for an OPEN claim after a crash.

    Completed or failed adjudications have already exposed the permanent holdout
    and are never releasable through this recovery API.
    """
    import duckdb

    from quant_loop_trader.data import DB_PATH, migrate_db

    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    row = con.execute(
        "SELECT state FROM holdout_claims WHERE experiment_id=?", [experiment_id]
    ).fetchone()
    if not row:
        con.close()
        return "no_claim"
    if row[0] != "CLAIMED":
        con.close()
        return f"refused:{row[0]}"
    con.execute("DELETE FROM holdout_claims WHERE experiment_id=?", [experiment_id])
    con.close()
    return "released:CLAIMED"


def _confirm_success_memory(experiment_id: str) -> None:
    import duckdb

    from quant_loop_trader.data import DB_PATH, migrate_db

    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    con.execute(
        "UPDATE research_memory SET lesson=?, confidence=LEAST(confidence + 0.15, 0.95), "
        "provenance_json=? WHERE experiment_id=? AND memory_type='success'",
        [
            "CONFIRMED by final hidden-holdout adjudication.",
            json.dumps({"confirmed_by": "adjudicate_holdout"}),
            experiment_id,
        ],
    )
    con.close()


def _load_holdout_frame(pq, cfg):
    """Compute causal features on full history, then isolate the hidden tail."""
    import polars as pl

    from quant_loop_trader.experiment import make_labels
    from quant_loop_trader.features import add_improved_features
    from quant_loop_trader.replay import ReplayEngine

    df = ReplayEngine(pq, ticker=cfg["ticker"]).get_snapshot(cfg["ticker"], cfg["end"])
    df = df.filter(
        pl.col("event_time") >= pl.lit(cfg["start"]).str.strptime(pl.Date, "%Y-%m-%d")
    )
    featured = add_improved_features(make_labels(df, cfg["horizon"]))
    hold = apply_holdout(featured, cfg["start"], cfg["end"], use_holdout=True)
    return hold.drop_nulls(subset=improved_feature_columns() + ["label"])


def _train_all_nonholdout(pq, cfg):
    import polars as pl

    from quant_loop_trader.experiment import make_labels
    from quant_loop_trader.features import add_improved_features, improved_feature_columns
    from quant_loop_trader.replay import ReplayEngine

    df = ReplayEngine(pq, ticker=cfg["ticker"]).get_snapshot(cfg["ticker"], cfg["end"])
    df = df.filter(
        pl.col("event_time") >= pl.lit(cfg["start"]).str.strptime(pl.Date, "%Y-%m-%d")
    )
    featured = add_improved_features(make_labels(df, cfg["horizon"]))
    clean = apply_holdout(featured, cfg["start"], cfg["end"], use_holdout=False)
    clean = clean.sort("event_time").slice(0, max(0, clean.height - cfg["horizon"]))
    clean = clean.drop_nulls(subset=improved_feature_columns() + ["label"])
    return clean.select(improved_feature_columns()).to_numpy(), clean["label"].to_numpy()
