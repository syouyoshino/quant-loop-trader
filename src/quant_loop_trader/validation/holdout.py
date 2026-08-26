"""Out-of-time holdout (Phase 7): the final segment of every window is PERMANENTLY
hidden from training/tuning/research splits. Only an explicit holdout evaluation
may touch it."""
from __future__ import annotations

from datetime import datetime

import numpy as np

from quant_loop_trader.features import improved_feature_columns

HOLDOUT_FRACTION = 0.15


def holdout_boundary(start: str, end: str) -> str:
    """ISO date where the hidden region begins for a [start, end] research window."""
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    days = int((e - s).days * HOLDOUT_FRACTION)
    return (e - __import__("datetime").timedelta(days=days)).date().isoformat()


def apply_holdout(df, start: str, end: str, use_holdout: bool):
    """use_holdout=False (research default): drop rows at/after the boundary so no
    model, feature selection, or tuning ever sees them.
    use_holdout=True (final evaluation ONLY): return just the hidden segment as test."""
    import polars as pl
    b = pl.lit(holdout_boundary(start, end)).str.strptime(pl.Date, "%Y-%m-%d")
    if use_holdout:
        return df.filter(pl.col("event_time") >= b)
    return df.filter(pl.col("event_time") < b)


def adjudicate_holdout(experiment_id: str) -> dict:
    """FINAL adjudication: opens the hidden segment exactly once for an ELIGIBLE
    experiment. Retrains on all non-holdout data, evaluates on the untouched tail,
    and promotes to champion only if it beats the majority-class base rate there.
    This is the only code path that may set model_registry status='champion'."""
    import json
    import logging

    import duckdb
    from quant_loop_trader.data import PROC_DIR, DB_PATH, migrate_db
    from quant_loop_trader.experiment import EXP_ROOT
    from quant_loop_trader.models.registry import build_model

    exp_dir = EXP_ROOT / experiment_id
    cfg = json.loads((exp_dir / "config.json").read_text())
    con = duckdb.connect(str(DB_PATH))
    row = con.execute("SELECT status FROM model_registry WHERE model_id=?", [f"{experiment_id}_improved"]).fetchone()
    con.close()
    if not row or row[0] != "eligible":
        return {"promoted": False, "reason": f"not_eligible:{row[0] if row else 'missing'}"}

    ticker, h = cfg["ticker"], cfg["horizon"]
    pq = PROC_DIR / f"{ticker}.parquet"
    fin = {}
    try:
        df = _load_holdout_frame(pq, cfg)
        # keep close for the economic gate; features for the classifier
        feats = df.select(improved_feature_columns() + ["label", "close"])
        Xte, yte = feats.select(_feature_names()).to_numpy(), feats["label"].to_numpy()
        # audit remediation: fit(X, y) — the previous call passed the (X, y) tuple
        # as X and None as y, guaranteeing a crash inside sklearn
        Xtr, ytr = _train_all_nonholdout(pq, cfg)
        m = build_model(cfg.get("model_type", "logistic"), seed=cfg["seed"]).fit(Xtr, ytr)
        ypred = m.predict(Xte)
        base = float(max(yte.mean(), 1 - yte.mean()))
        acc = float((ypred == yte).mean())
        # predeclared ECONOMIC objective (audit round-2): classification accuracy
        # alone never promotes — the hidden tail must also clear costs and beat
        # buy-and-hold on risk-adjusted net terms
        from quant_loop_trader.evaluation import evaluate
        prices_h = feats["close"].to_numpy()
        try:
            prob = m.predict_proba(Xte)
        except Exception:
            prob = ypred.astype(float)
        fin = evaluate(yte, ypred, prob, prices_h, horizon=h)
        # audit H2: compounded WEALTH, not arithmetic return sums.
        # +60%/-50% sums to +10% but loses 20% of capital — only the compounded
        # figure reflects what the capital actually did.
        economic_gate = (fin["cumulative_return_strategy"] > 0
                         and fin["sharpe_strategy"] >= fin["sharpe_benchmark"])
        # audit B/H-round3: significance must come from HOLDOUT predictions, not
        # the research split. One-sided binomial vs majority base rate on the
        # hidden tail itself, via the ONE canonical significance function.
        from quant_loop_trader.core import significance as _sig
        sig = _sig(yte, np.asarray(ypred), horizon=h)
        sig_ok = sig.passed and sig.n_effective >= 10
        # NOTE: research-split statistical approval was already required for
        # ELIGIBILITY; here ONLY holdout evidence decides (audit invariant 8).
        promoted = bool(acc > base and economic_gate and sig_ok)
    except Exception as e:
        # fail closed: a broken adjudication can never promote (audit cycle-2 D)
        import traceback
        logger = logging.getLogger(__name__)
        logger.error(f"adjudication failed: {e}")
        result = {"promoted": False, "reason": f"adjudication_error:{str(e)[:150]}",
                  "economic_gate": fin or None,
                  "traceback_tail": traceback.format_exc()[-300:]}
        (exp_dir / "holdout_report.json").write_text(json.dumps(result, indent=2))
        return result
    result = {"promoted": promoted, "holdout_accuracy": acc,
              "base_rate": base, "n_holdout": int(len(yte)),
              "economic_gate": {"compounded_net_return": fin["cumulative_return_strategy"],
                                "sharpe_strategy": fin["sharpe_strategy"],
                                "sharpe_benchmark": fin["sharpe_benchmark"]}}
    (exp_dir / "holdout_report.json").write_text(json.dumps(result, indent=2))
    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    new_status = "champion" if promoted else "rejected"
    con.execute("UPDATE model_registry SET status=? WHERE model_id=?", [new_status, f"{experiment_id}_improved"])
    con.close()
    # audit H6 + round-3 branch fix: confirm on promotion, correct on rejection —
    # the previous unconditional correct→confirm sequence left real champions
    # recorded as failures because their success rows no longer existed to confirm.
    from quant_loop_trader.agents import _correct_success_memories_for
    if promoted:
        _confirm_success_memory(experiment_id)
    else:
        _correct_success_memories_for(experiment_id)
    return result


def _confirm_success_memory(experiment_id: str) -> None:
    import duckdb
    from quant_loop_trader.data import DB_PATH, migrate_db
    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    con.execute(
        "UPDATE research_memory SET lesson=?, confidence=LEAST(confidence + 0.15, 0.95), "
        "provenance_json=? WHERE experiment_id=? AND memory_type='success'",
        ["CONFIRMED by final hidden-holdout adjudication.",
         json.dumps({"confirmed_by": "adjudicate_holdout"}), experiment_id],
    )
    con.close()


def _load_holdout_frame(pq, cfg):
    """Audit M1: compute causal features over the FULL PIT history FIRST, then
    isolate holdout rows — pre-holdout history legitimately feeds the feature
    windows of early holdout dates (no leakage: everything is still causal)."""
    import polars as pl
    from quant_loop_trader.replay import ReplayEngine
    from quant_loop_trader.experiment import make_labels
    from quant_loop_trader.features import add_improved_features
    df = ReplayEngine(pq).get_snapshot(cfg["ticker"], cfg["end"])
    df = df.filter(pl.col("event_time") >= pl.lit(cfg["start"]).str.strptime(pl.Date, "%Y-%m-%d"))
    featured = add_improved_features(make_labels(df, cfg["horizon"]))
    hold = apply_holdout(featured, cfg["start"], cfg["end"], use_holdout=True)
    return hold.drop_nulls(subset=improved_feature_columns() + ["label"])


def _improved_cols(df):
    from quant_loop_trader.features import improved_feature_columns
    return df.select(improved_feature_columns() + ["label"])


def _feature_names():
    from quant_loop_trader.features import improved_feature_columns
    return improved_feature_columns()


def _train_all_nonholdout(pq, cfg):
    import polars as pl
    from quant_loop_trader.replay import ReplayEngine
    from quant_loop_trader.experiment import make_labels
    from quant_loop_trader.features import add_improved_features, improved_feature_columns
    df = ReplayEngine(pq).get_snapshot(cfg["ticker"], cfg["end"])
    df = df.filter(pl.col("event_time") >= pl.lit(cfg["start"]).str.strptime(pl.Date, "%Y-%m-%d"))
    # features computed on the full history first (M1), then holdout rows removed
    featured = add_improved_features(make_labels(df, cfg["horizon"]))
    clean = apply_holdout(featured, cfg["start"], cfg["end"], use_holdout=False)
    # EMBARGO (audit round-3 CRITICAL): drop the last h pre-boundary rows whose
    # labels were computed from prices INSIDE the hidden holdout. Features keep
    # their full causal warmup; outcomes that peek into the holdout do not.
    clean = clean.sort("event_time").slice(0, max(0, clean.height - cfg["horizon"]))
    clean = clean.drop_nulls(subset=improved_feature_columns() + ["label"])
    return clean.select(improved_feature_columns()).to_numpy(), clean["label"].to_numpy()


