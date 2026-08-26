"""Single-command research loop: acquire once → snapshot → research → store → reproduce."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from quant_loop_trader.data import (
    DB_PATH,
    PROC_DIR,
    dataset_metadata,
    fetch_ohlcv,
    migrate_db,
    upsert_dataset,
)
from quant_loop_trader.evaluation import autopsy, evaluate, time_split
from quant_loop_trader.features import (
    add_features,
    add_improved_features,
    feature_columns,
    improved_feature_columns,
)
from quant_loop_trader.replay import ReplayEngine
from quant_loop_trader.research_memory import (
    duplicate_risk,
    record_outcome,
    register_features,
    register_model,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(os.environ.get("QLT_ROOT", Path.cwd()))
EXP_ROOT = ROOT / "data" / "experiments"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_labels(df: pl.DataFrame, horizon: int = 5) -> pl.DataFrame:
    """Add label = 1 if close[t+h]/close[t]-1 > 0, aligned at t."""
    df = df.sort("event_time")
    df = df.with_columns(
        (pl.col("close").shift(-horizon) / pl.col("close") - 1).alias("fwd_ret"),
    )
    return df.with_columns((pl.col("fwd_ret") > 0).cast(pl.Int8).alias("label"))


def build_train_test(ticker: str, start: str, end: str, horizon: int, feature_fn,
                     feat_cols: list[str], parquet_path: Path | None = None):
    """Build the canonical research split from one immutable input parquet.

    Acquisition is deliberately outside this function. ``parquet_path`` should be
    the experiment's content-addressed dataset snapshot. The cache fallback remains
    only for compatibility with direct callers/tests that have not acquired data.
    """
    pq = Path(parquet_path) if parquet_path else PROC_DIR / f"{ticker}.parquet"
    if not pq.exists():
        df_raw, _ = fetch_ohlcv(ticker, start, end)
        pq.parent.mkdir(parents=True, exist_ok=True)
        df_raw.write_parquet(str(pq))
    df = ReplayEngine(pq, ticker=ticker).get_snapshot(ticker, end)
    df = df.filter(pl.col("event_time") >= pl.lit(start).str.strptime(pl.Date, "%Y-%m-%d"))
    from quant_loop_trader.validation.holdout import apply_holdout
    df = apply_holdout(df, start, end, use_holdout=False)
    df = make_labels(df, horizon)
    df_clean = feature_fn(df).drop_nulls(subset=feat_cols + ["label"])
    if df_clean.height < 100:
        raise ValueError(f"not enough rows after feature cleaning: {df_clean.height}")
    train, test = time_split(df_clean, 0.7, purge=horizon)
    return train, test


def train_evaluate_from(train: pl.DataFrame, test: pl.DataFrame, feat_cols: list[str],
                        horizon: int, seed: int) -> dict:
    """Train on prebuilt frames and evaluate the hidden research test split."""
    from quant_loop_trader.models.registry import build_model

    _seed_all(seed)
    X_train = train.select(feat_cols).to_numpy()
    y_train = train["label"].to_numpy()
    X_test = test.select(feat_cols).to_numpy()
    y_test = test["label"].to_numpy()
    prices_test = test["close"].to_numpy()

    model = build_model("logistic", seed=seed)
    model.fit(
        X_train,
        y_train,
        train_period=(str(train["event_time"].min()), str(train["event_time"].max())),
    )
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)
    except Exception:
        y_prob = y_pred.astype(float)

    metrics = evaluate(y_test, y_pred, y_prob, prices_test, horizon)
    # ONE significance implementation. This p-value is also the candidate-family
    # FDR input, so it must use the same non-overlapping one-sided test everywhere.
    from quant_loop_trader.core import significance
    metrics["stat_pvalue"] = significance(
        y_test, y_pred, horizon=horizon
    ).pvalue
    err = autopsy(test, y_test, y_pred)

    pred_df = pl.DataFrame({
        "event_time": test["event_time"],
        "available_time": test["available_time"],
        "close": test["close"],
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    })
    return {
        "metrics": metrics,
        "error_analysis": err,
        "pred_df": pred_df,
        "train_n": train.height,
        "test_n": test.height,
        "model": model,
    }


_PIPELINE_REVISION = 2


def _pipeline_version() -> int:
    return _PIPELINE_REVISION


def _code_version() -> str:
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_experiment(ticker: str = "SPY", horizon: int = 5,
                   start: str = "2018-01-01", end: str = "2024-12-31",
                   seed: int = 42, source_snapshot: str | Path | None = None,
                   parent_experiment_id: str | None = None) -> dict:
    """Run one experiment.

    Acquisition happens at most once. Immediately afterwards the exact dataframe is
    stored as a content-addressed snapshot, and every research computation consumes
    that snapshot rather than the mutable ticker cache.
    """
    from quant_loop_trader.core import ExperimentSpec

    spec = ExperimentSpec(
        ticker=ticker,
        start=start,
        end=end,
        horizon=horizon,
        seed=seed,
        pipeline_version=_pipeline_version(),
    )
    config_hash = spec.fingerprint()[:8]
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    exp_id = f"{day}_{ticker}_{horizon}d_{config_hash}"
    exp_dir = EXP_ROOT / exp_id
    run_suffix = 0
    while exp_dir.exists():
        run_suffix += 1
        exp_id = f"{day}_{ticker}_{horizon}d_{config_hash}_r{run_suffix}"
        exp_dir = EXP_ROOT / exp_id
    exp_dir.mkdir(parents=True, exist_ok=False)

    # 1. acquire or reuse an already-sealed source snapshot
    if source_snapshot is None:
        df_raw, source = fetch_ohlcv(ticker, start, end)
    else:
        source_path = Path(source_snapshot)
        if not source_path.exists():
            raise FileNotFoundError(f"source snapshot not found: {source_path}")
        df_raw = pl.read_parquet(str(source_path))
        source = "snapshot_reproduction"

    meta = dataset_metadata(
        df_raw,
        ticker,
        source,
        extra_provenance={"start": start, "end": end},
    )
    upsert_dataset(meta)

    snap_dir = ROOT / "data" / "datasets"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"{meta['dataset_id']}.parquet"
    if not snap_path.exists():
        df_raw.write_parquet(str(snap_path))

    # 2. all post-acquisition research uses the immutable snapshot
    train_b, test_b = build_train_test(
        ticker, start, end, horizon, add_features, feature_columns(), snap_path
    )
    baseline = train_evaluate_from(train_b, test_b, feature_columns(), horizon, seed)
    train_i, test_i = build_train_test(
        ticker, start, end, horizon, add_improved_features,
        improved_feature_columns(), snap_path
    )
    improved = train_evaluate_from(
        train_i, test_i, improved_feature_columns(), horizon, seed
    )

    periods = {
        "train_period": [str(train_b["event_time"].min()), str(train_b["event_time"].max())],
        "test_period": [str(test_b["event_time"].min()), str(test_b["event_time"].max())],
        "train_rows": train_b.height,
        "test_rows": test_b.height,
    }

    b_acc = baseline["metrics"]["accuracy"]
    i_acc = improved["metrics"]["accuracy"]
    b_sharpe = baseline["metrics"]["sharpe_strategy"]
    i_sharpe = improved["metrics"]["sharpe_strategy"]
    improvement = i_acc - b_acc
    if i_acc > b_acc and i_sharpe >= b_sharpe:
        decision = "KEEP"
    elif i_acc > b_acc:
        decision = "IMPROVE"
    else:
        decision = "REJECT"

    hypothesis = (
        "Adding volatility regime classification should improve momentum prediction "
        "because trend persistence differs across volatility environments."
    )
    economic_reasoning = (
        "High-vol regimes reflect noise/mean-reversion and crowded positioning; "
        "low-vol regimes allow trend persistence due to gradual information diffusion "
        "and institutional herding."
    )
    research_question = (
        f"Does volatility regime filtering improve {horizon}-day directional "
        f"prediction for {ticker}?"
    )
    dup = duplicate_risk(hypothesis)
    if dup["should_warn"]:
        logger.warning(json.dumps({
            "event": "duplicate_risk",
            "similar_failures": dup["similar_failures"],
            "hypothesis": hypothesis[:60],
        }))

    config = {
        "spec_fingerprint": spec.fingerprint(),
        "pipeline_version": _pipeline_version(),
        "ticker": ticker,
        "horizon": horizon,
        "start": start,
        "end": end,
        "seed": seed,
        "dataset_id": meta["dataset_id"],
        "feature_version_baseline": "v1-ret1-ret5-ma10-vol10-rsi14",
        "feature_version_improved": "v1+vol_regime_ret5_x_vol10",
        "model_version": "sklearn-LogReg-C1.0-scaled",
        "snapshot_definition": meta["snapshot_definition"],
        "dataset_checksum": meta["checksum"],
        "dataset_snapshot": str(snap_path),
        **periods,
    }
    report = {
        "experiment_id": exp_id,
        "research_question": research_question,
        "hypothesis": hypothesis,
        "economic_reasoning": economic_reasoning,
        "creator_agent": "baseline_researcher",
        "research_priority_score": 0.8,
        "experiment_design": (
            f"time-split 70/30, LogisticRegression, features baseline "
            f"{feature_columns()} vs improved {improved_feature_columns()}"
        ),
        "expected_outcome": "improved accuracy +0.02 and Sharpe non-degrading",
        "success_criteria": "improved accuracy > baseline and Sharpe >= baseline",
        "failure_condition": "no accuracy lift or Sharpe degrades",
        "dataset_version": meta["version"],
        "dataset_id": meta["dataset_id"],
        "snapshot_definition": meta["snapshot_definition"],
        "data_dependencies": [str(snap_path)],
        "provenance": meta["provenance_json"],
        "model_version": config["model_version"],
        "feature_version": config["feature_version_improved"],
        "parameters": {
            "horizon": horizon,
            "seed": seed,
            "model": "LogisticRegression",
            "scaler": "StandardScaler",
        },
        "prediction_timestamp": end,
        "baseline_metrics": baseline["metrics"],
        "improved_metrics": improved["metrics"],
        "benchmark_result": {
            "cumulative_return_benchmark": baseline["metrics"]["cumulative_return_benchmark"],
            "sharpe_benchmark": baseline["metrics"]["sharpe_benchmark"],
        },
        "volatility": {
            "strategy": improved["metrics"]["volatility_strategy"],
            "benchmark": improved["metrics"]["volatility_benchmark"],
        },
        "drawdown": {"strategy": improved["metrics"]["max_drawdown_strategy"]},
        "transaction_cost_adjusted_result": improved["metrics"]["transaction_cost_adj_return"],
        "error_analysis": baseline["error_analysis"],
        "improved_error_analysis": improved["error_analysis"],
        "root_cause_analysis": (
            f"baseline regime accuracies {baseline['error_analysis']}, "
            f"improvement delta {improvement:.4f}"
        ),
        "improvement_attempt": "added vol10 interaction terms ret5_x_vol10, ret5_div_vol10",
        "decision": decision,
        "improvement_delta_accuracy": float(improvement),
        "improvement_delta_sharpe": float(i_sharpe - b_sharpe),
        "parent_experiment_id": parent_experiment_id,
        "research_branch": "research/mvrs",
        "mutation_reason": "volatility regime hypothesis",
        "final_result": decision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "stat_pvalue": baseline["metrics"].get("stat_pvalue"),
        "candidate_stat_pvalue": improved["metrics"].get("stat_pvalue"),
        "reproducibility": {
            "seed": seed,
            "checksum": meta["checksum"],
            "code_version": _code_version(),
        },
    }

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    baseline["pred_df"].write_parquet(str(exp_dir / "predictions_baseline.parquet"))
    improved["pred_df"].write_parquet(str(exp_dir / "predictions_improved.parquet"))
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    (exp_dir / "metrics.json").write_text(json.dumps({
        "baseline": baseline["metrics"],
        "improved": improved["metrics"],
    }, indent=2))
    (exp_dir / "report.json").write_text(json.dumps(report, indent=2))
    (exp_dir / "predictions.lock").write_text(json.dumps({
        "predictions_baseline.parquet": _sha(exp_dir / "predictions_baseline.parquet"),
        "predictions_improved.parquet": _sha(exp_dir / "predictions_improved.parquet"),
        "config.json": _sha(exp_dir / "config.json"),
        "metrics.json": _sha(exp_dir / "metrics.json"),
        "report.json": _sha(exp_dir / "report.json"),
        "dataset_snapshot_sha256": _sha(snap_path),
        "locked_at": report["created_at"],
    }, indent=2))

    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    for suffix, res, feat_ver in [
        ("baseline", baseline, "v1"),
        ("improved", improved, "v1+vol_regime"),
    ]:
        eid = f"{exp_id}_{suffix}"
        con.execute(
            "INSERT OR REPLACE INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp, TRUE)",
            [
                eid,
                meta["dataset_id"],
                ticker,
                horizon,
                "v1",
                hypothesis if suffix == "improved" else "baseline momentum",
                economic_reasoning if suffix == "improved" else "momentum persists",
                research_question,
                config["model_version"],
                feat_ver,
                seed,
                json.dumps(config),
                json.dumps(res["metrics"]),
                decision if suffix == "improved" else "candidate",
                exp_id if suffix == "improved" else None,
                json.dumps({
                    "suffix": suffix,
                    "train_n": res["train_n"],
                    "test_n": res["test_n"],
                }),
            ],
        )
    con.close()

    register_features([
        {"feature_id": "ret_1", "formula": "shift(close/close.shift(1)-1, 1)"},
        {"feature_id": "ret_5", "formula": "shift(close/close.shift(5)-1, 1)"},
        {"feature_id": "ma10_gap", "formula": "shift((close-sma10)/close, 1)"},
        {"feature_id": "vol10", "formula": "shift(std(ret_1,10), 1)"},
        {"feature_id": "rsi14", "formula": "shift(rsi(14, simple-mean gains/losses), 1)"},
        {
            "feature_id": "ret5_x_vol10",
            "formula": "ret_5(shifted) * vol10(shifted) — inputs are already lagged; no extra shift",
            "failure_conditions": "degenerate when vol10 near zero",
        },
        {
            "feature_id": "ret5_div_vol10",
            "formula": "ret_5(shifted) / (vol10(shifted)+1e-9) — inputs are already lagged; no extra shift",
            "failure_conditions": "unstable at low vol",
        },
    ])

    register_model({
        "model_id": f"{exp_id}_baseline",
        "training_data_version": meta["dataset_id"],
        "feature_version": config["feature_version_baseline"],
        "parameters_json": json.dumps(report["parameters"]),
        "performance_history_json": json.dumps(baseline["metrics"]),
        "failure_modes": "majority-class collapse when signal weak (all-positive predictions)",
        "research_lineage": f"root:{exp_id}",
        "status": "candidate",
    })
    register_model({
        "model_id": f"{exp_id}_improved",
        "parent_model_id": f"{exp_id}_baseline",
        "training_data_version": meta["dataset_id"],
        "feature_version": config["feature_version_improved"],
        "parameters_json": json.dumps(report["parameters"]),
        "performance_history_json": json.dumps(improved["metrics"]),
        "failure_modes": "same as baseline; interaction terms add no lift in tested regime",
        "research_lineage": f"momentum→vol_regime_filter:{exp_id}",
        "status": "rejected" if decision == "REJECT" else "candidate",
    })

    record_outcome(report)
    logger.info(json.dumps({
        "event": "experiment_done",
        "experiment_id": exp_id,
        "decision": decision,
        "delta_acc": float(improvement),
        "path": str(exp_dir),
    }))
    return report


def reproduce(experiment_id: str, tolerance: float = 1e-9) -> dict:
    """Reproduce from the original verified immutable dataset snapshot."""
    from quant_loop_trader.bundle import ExperimentBundle

    bundle = ExperimentBundle.open_verified(experiment_id, EXP_ROOT)
    original = bundle.report
    report = run_experiment(
        ticker=bundle.config["ticker"],
        horizon=bundle.config["horizon"],
        start=bundle.config["start"],
        end=bundle.config["end"],
        seed=bundle.config["seed"],
        source_snapshot=bundle.dataset_snapshot,
        parent_experiment_id=experiment_id,
    )
    acc_delta = abs(
        report["improved_metrics"]["accuracy"]
        - original["improved_metrics"]["accuracy"]
    )
    reproduced = acc_delta <= tolerance and report["decision"] == original["decision"]
    report["reproduction_check"] = {
        "reproduced": reproduced,
        "accuracy_delta": acc_delta,
        "tolerance": tolerance,
    }

    # Persist the reproduction result and reseal the report hash. The child was
    # already created with parent lineage, so the durable bundle is self-contained.
    child_dir = EXP_ROOT / report["experiment_id"]
    report_path = child_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    lock_path = child_dir / "predictions.lock"
    lock = json.loads(lock_path.read_text())
    lock["report.json"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    lock_path.write_text(json.dumps(lock, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="MVP Research loop — single command")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reproduce", type=str, default=None, help="experiment_id to reproduce")
    p.add_argument("--validate", type=str, default=None, help="experiment_id to run validation gate on")
    args = p.parse_args()
    if args.validate:
        from quant_loop_trader.agents import validate_experiment
        verdict = validate_experiment(args.validate)
        print(json.dumps(verdict, indent=2))
        return
    if args.reproduce:
        report = reproduce(args.reproduce)
    else:
        report = run_experiment(args.ticker, args.horizon, args.start, args.end, args.seed)
    print(json.dumps({
        "experiment_id": report["experiment_id"],
        "decision": report["decision"],
        "delta_acc": report["improvement_delta_accuracy"],
    }, indent=2))


if __name__ == "__main__":
    main()


def run_horizons(ticker: str = "SPY", horizons: list[int] | None = None,
                 start: str = "2018-01-01", end: str = "2024-12-31",
                 seed: int = 42) -> list[dict]:
    from quant_loop_trader.models.prediction import SUPPORTED_HORIZONS
    horizons = horizons or SUPPORTED_HORIZONS
    unsupported = [h for h in horizons if h not in SUPPORTED_HORIZONS]
    if unsupported:
        raise ValueError(f"unsupported horizons {unsupported}; supported: {SUPPORTED_HORIZONS}")
    return [
        run_experiment(ticker=ticker, horizon=h, start=start, end=end, seed=seed)
        for h in horizons
    ]


def get_experiment(experiment_id: str) -> dict:
    exp_dir = EXP_ROOT / experiment_id
    return json.loads((exp_dir / "report.json").read_text())


def list_experiments(limit: int = 100) -> list[dict]:
    import duckdb as _d
    migrate_db()
    con = _d.connect(str(DB_PATH))
    rows = con.execute(
        "SELECT experiment_id, ticker, horizon_days, decision, created_at FROM experiments "
        "WHERE authoritative ORDER BY created_at DESC LIMIT ?", [limit]
    ).fetchall()
    con.close()
    cols = ["experiment_id", "ticker", "horizon_days", "decision", "created_at"]
    out = []
    seen = set()
    for r in rows:
        d = dict(zip(cols, r))
        base = d["experiment_id"].replace("_baseline", "").replace("_improved", "")
        if base in seen:
            continue
        seen.add(base)
        d["experiment_id"] = base
        out.append(d)
    return out


def compare_experiments(experiment_ids: list[str]) -> pl.DataFrame:
    rows = []
    for eid in experiment_ids:
        r = get_experiment(eid)
        m = r["improved_metrics"]
        rows.append({
            "experiment_id": eid,
            "decision": r["decision"],
            "horizon": r["config"]["horizon"],
            **{k: round(v, 6) for k, v in m.items() if isinstance(v, (int, float))},
        })
    return pl.DataFrame(rows)
