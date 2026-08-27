"""Permanent out-of-time holdout and final adjudication."""
from __future__ import annotations

from datetime import datetime

import numpy as np

from quant_loop_trader.features import improved_feature_columns

HOLDOUT_FRACTION = 0.15


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


def adjudicate_holdout(experiment_id: str) -> dict:
    """Evaluate an eligible model exactly once on its sealed dataset snapshot."""
    import json
    import logging

    import duckdb

    from quant_loop_trader.bundle import BundleIntegrityError, ExperimentBundle
    from quant_loop_trader.data import DB_PATH, migrate_db
    from quant_loop_trader.experiment import EXP_ROOT
    from quant_loop_trader.models.registry import build_model

    exp_dir = EXP_ROOT / experiment_id
    try:
        bundle = ExperimentBundle.open_verified(experiment_id, EXP_ROOT)
    except BundleIntegrityError as exc:
        result = {"promoted": False, "reason": f"bundle_integrity:{str(exc)[:150]}"}
        if exp_dir.exists():
            (exp_dir / "holdout_report.json").write_text(json.dumps(result, indent=2))
        return result

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
            fin["cumulative_return_strategy"] > 0
            and fin["sharpe_strategy"] >= fin["sharpe_benchmark"]
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
        (exp_dir / "holdout_report.json").write_text(json.dumps(result, indent=2))
        _close_holdout_claim(experiment_id, "FAILED", result)
        return result

    result = {
        "promoted": promoted,
        "holdout_accuracy": acc,
        "base_rate": base,
        "n_holdout": int(len(yte)),
        "economic_gate": {
            "compounded_net_return": fin["cumulative_return_strategy"],
            "sharpe_strategy": fin["sharpe_strategy"],
            "sharpe_benchmark": fin["sharpe_benchmark"],
        },
        # the full evaluate() output on the hidden holdout — drawdown, Sortino,
        # VaR, turnover and the rest were already computed and then discarded
        "holdout_metrics": fin,
    }
    (exp_dir / "holdout_report.json").write_text(json.dumps(result, indent=2))

    from quant_loop_trader.core import LifecycleEvidence, final_state

    new_status = final_state(LifecycleEvidence(
        research_screen=bundle.report["decision"],
        validation="PASS",
        holdout="PASS" if promoted else "FAIL",
    ))
    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    con.execute(
        "UPDATE model_registry SET status=? WHERE model_id=?",
        [new_status, f"{experiment_id}_improved"],
    )
    con.close()

    _close_holdout_claim(experiment_id, "COMPLETE", result)

    from quant_loop_trader.agents import _correct_success_memories_for

    if promoted:
        _confirm_success_memory(experiment_id)
    else:
        _correct_success_memories_for(experiment_id)
    return result


def _claim_holdout(experiment_id: str) -> dict | None:
    """Take the one-shot lock before the holdout is exposed.

    Returns None when the claim is ours, or the refusal to return otherwise.
    A CLAIMED row with no COMPLETE means a previous attempt died after the
    holdout was read: fail closed and require `release_holdout_claim`.
    """
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
    except duckdb.ConstraintException:  # a concurrent adjudication won the race
        return {"promoted": False, "reason": "holdout_already_consumed:CLAIMED"}
    finally:
        con.close()
    return None


def _close_holdout_claim(experiment_id: str, state: str, result: dict) -> None:
    import json

    import duckdb

    from quant_loop_trader.data import DB_PATH

    con = duckdb.connect(str(DB_PATH))
    con.execute(
        "UPDATE holdout_claims SET state=?, completed_at=current_timestamp, "
        "promoted=?, result_json=? WHERE experiment_id=?",
        [state, bool(result.get("promoted")), json.dumps(result), experiment_id],
    )
    con.close()


def release_holdout_claim(experiment_id: str) -> str:
    """Explicit human recovery after a crash mid-adjudication.

    Deliberately not called by any automated path: re-running a holdout is a
    scientific decision, not a retry.
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
    con.execute("DELETE FROM holdout_claims WHERE experiment_id=?", [experiment_id])
    con.close()
    return f"released:{row[0]}"


def _confirm_success_memory(experiment_id: str) -> None:
    import json

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
