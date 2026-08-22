"""Single-command MVP research loop: fetch→PIT→train→predict→evaluate→autopsy→improve→compare→store→reproduce."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import numpy as np
import duckdb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from quant_loop_trader.data import fetch_ohlcv, save_parquet, dataset_metadata, upsert_dataset, migrate_db, DB_PATH, PROC_DIR
from quant_loop_trader.replay import ReplayEngine
from quant_loop_trader.features import add_features, add_improved_features, feature_columns, improved_feature_columns
from quant_loop_trader.evaluation import time_split, evaluate, autopsy

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "data" / "experiments"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_labels(df: pl.DataFrame, horizon: int = 5) -> pl.DataFrame:
    """Add label = 1 if close[t+h]/close[t]-1 >0 else 0. Keep event_time aligned at t."""
    df = df.sort("event_time")
    df = df.with_columns(
        (pl.col("close").shift(-horizon) / pl.col("close") - 1).alias("fwd_ret"),
    )
    df = df.with_columns(
        (pl.col("fwd_ret") > 0).cast(pl.Int8).alias("label"),
    )
    return df


def train_evaluate(df: pl.DataFrame, feature_fn, feat_cols: list[str], horizon: int, seed: int) -> dict:
    _seed_all(seed)
    df = make_labels(df, horizon)
    # drop rows where label or features are null
    df_feat = feature_fn(df)
    # need at least one feature + label
    df_clean = df_feat.drop_nulls(subset=feat_cols + ["label"])
    if df_clean.height < 100:
        raise ValueError(f"not enough rows after feature cleaning: {df_clean.height}")
    train, test = time_split(df_clean, 0.7)
    # hidden future: test labels not seen during training — evaluation owns them
    X_train = train.select(feat_cols).to_numpy()
    y_train = train["label"].to_numpy()
    X_test = test.select(feat_cols).to_numpy()
    y_test = test["label"].to_numpy()
    prices_test = test["close"].to_numpy()

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=seed))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    try:
        y_prob = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = y_pred.astype(float)

    metrics = evaluate(y_test, y_pred, y_prob, prices_test)
    err = autopsy(test, y_test, y_pred)

    # predictions frame for storage (only test)
    pred_df = pl.DataFrame({
        "event_time": test["event_time"],
        "available_time": test["available_time"],
        "close": test["close"],
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    })
    return {"metrics": metrics, "error_analysis": err, "pred_df": pred_df, "train_n": train.height, "test_n": test.height, "model": pipe}


def run_experiment(ticker: str = "SPY", horizon: int = 5, start: str = "2018-01-01", end: str = "2024-12-31", seed: int = 42) -> dict:
    exp_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d')}_{ticker}_{horizon}d_{hashlib.sha256(f'{ticker}{horizon}{seed}{start}{end}'.encode()).hexdigest()[:8]}"
    exp_dir = EXP_ROOT / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. data
    df_raw = fetch_ohlcv(ticker, start, end)
    meta = dataset_metadata(df_raw, ticker, "tiingo" if "TIINGO_API_KEY" in str(fetch_ohlcv.__code__) else "fixture")
    # determine source more accurately
    import os
    source = "tiingo" if os.getenv("TIINGO_API_KEY", "").strip() else "fixture"
    meta["source"] = source
    meta["provenance_json"] = json.dumps({"source": source, "ticker": ticker, "start": start, "end": end})
    # save parquet already done in fetch, ensure replay engine uses it
    parquet_path = PROC_DIR / f"{ticker}.parquet"
    if not parquet_path.exists():
        save_parquet(df_raw, parquet_path)
    upsert_dataset(meta)

    # 2. replay verification
    engine = ReplayEngine(parquet_path)
    # sanity: snapshot at end should equal full history
    snap_full = engine.get_snapshot(ticker, str(df_raw["event_time"].max()))
    assert snap_full.height == df_raw.height, "replay full snapshot mismatch"

    df_snapshot = engine.get_snapshot(ticker, end)  # full available at end
    # For PIT correctness we use df_raw directly (already PIT-filtered by fetch). In L1 available==event.

    # 3. baseline
    baseline = train_evaluate(df_snapshot, add_features, feature_columns(), horizon, seed)
    # 4. improved
    improved = train_evaluate(df_snapshot, add_improved_features, improved_feature_columns(), horizon, seed)

    # 5. compare
    b_acc = baseline["metrics"]["accuracy"]
    i_acc = improved["metrics"]["accuracy"]
    b_sharpe = baseline["metrics"]["sharpe_strategy"]
    i_sharpe = improved["metrics"]["sharpe_strategy"]
    improvement = i_acc - b_acc
    # decision: KEEP if both accuracy and sharpe improve; else REJECT; failed still stored
    if i_acc > b_acc and i_sharpe >= b_sharpe:
        decision = "KEEP"
    elif i_acc > b_acc:
        decision = "IMPROVE"
    else:
        decision = "REJECT"
    # improvement hypothesis per spec
    hypothesis = "Adding volatility regime classification should improve momentum prediction because trend persistence differs across volatility environments."
    economic_reasoning = "High-vol regimes reflect noise/mean-reversion and crowded positioning; low-vol regimes allow trend persistence due to gradual information diffusion and institutional herding."
    research_question = f"Does volatility regime filtering improve {horizon}-day directional prediction for {ticker}?"
    # dataset version etc
    config = {
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
    }
    report = {
        # Research
        "experiment_id": exp_id,
        "research_question": research_question,
        "hypothesis": hypothesis,
        "economic_reasoning": economic_reasoning,
        "creator_agent": "baseline_researcher",
        "research_priority_score": 0.8,
        "experiment_design": f"time-split 70/30, LogisticRegression, features baseline {feature_columns()} vs improved {improved_feature_columns()}",
        "expected_outcome": "improved accuracy +0.02 and Sharpe non-degrading",
        "success_criteria": "improved accuracy > baseline and Sharpe >= baseline",
        "failure_condition": "no accuracy lift or Sharpe degrades",
        # Data
        "dataset_version": meta["version"],
        "dataset_id": meta["dataset_id"],
        "snapshot_definition": meta["snapshot_definition"],
        "data_dependencies": [str(parquet_path)],
        "provenance": meta["provenance_json"],
        # Model
        "model_version": config["model_version"],
        "feature_version": config["feature_version_improved"],
        "parameters": {"horizon": horizon, "seed": seed, "model": "LogisticRegression", "scaler": "StandardScaler"},
        # Prediction
        "prediction_timestamp": end,
        # Outcome
        "baseline_metrics": baseline["metrics"],
        "improved_metrics": improved["metrics"],
        "benchmark_result": {"cumulative_return_benchmark": baseline["metrics"]["cumulative_return_benchmark"], "sharpe_benchmark": baseline["metrics"]["sharpe_benchmark"]},
        "volatility": {"strategy": improved["metrics"]["volatility_strategy"], "benchmark": improved["metrics"]["volatility_benchmark"]},
        "drawdown": {"strategy": improved["metrics"]["max_drawdown_strategy"]},
        "transaction_cost_adjusted_result": improved["metrics"]["transaction_cost_adj_return"],
        # Analysis
        "error_analysis": baseline["error_analysis"],
        "improved_error_analysis": improved["error_analysis"],
        "root_cause_analysis": f"baseline regime accuracies {baseline['error_analysis']}, improvement delta {improvement:.4f}",
        "improvement_attempt": "added vol10 interaction terms ret5_x_vol10, ret5_div_vol10",
        # Decision
        "decision": decision,
        "improvement_delta_accuracy": float(improvement),
        "improvement_delta_sharpe": float(i_sharpe - b_sharpe),
        # lineage
        "parent_experiment_id": None,
        "research_branch": "research/mvrs",
        "mutation_reason": "volatility regime hypothesis",
        "final_result": decision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "reproducibility": {"seed": seed, "checksum": meta["checksum"]},
    }

    # 6. store predictions
    baseline["pred_df"].write_parquet(str(exp_dir / "predictions_baseline.parquet"))
    improved["pred_df"].write_parquet(str(exp_dir / "predictions_improved.parquet"))
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    (exp_dir / "metrics.json").write_text(json.dumps({"baseline": baseline["metrics"], "improved": improved["metrics"]}, indent=2))
    (exp_dir / "report.json").write_text(json.dumps(report, indent=2))
    # ledger jsonl
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    with open(EXP_ROOT / "experiments.jsonl", "a") as f:
        f.write(json.dumps({"experiment_id": exp_id, "decision": decision, "delta_acc": float(improvement), "created_at": report["created_at"]}) + "\n")

    # 7. DuckDB insert
    migrate_db()
    con = duckdb.connect(str(DB_PATH))
    # baseline as candidate, improved as champion if KEEP
    for suffix, res, feat_ver in [("baseline", baseline, "v1"), ("improved", improved, "v1+vol_regime")]:
        eid = f"{exp_id}_{suffix}"
        con.execute(
            "INSERT OR REPLACE INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)",
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
                json.dumps({"suffix": suffix, "train_n": res["train_n"], "test_n": res["test_n"]}),
            ],
        )
    con.close()

    logger.info(json.dumps({"event": "experiment_done", "experiment_id": exp_id, "decision": decision, "delta_acc": float(improvement), "path": str(exp_dir)}))
    return report


def reproduce(experiment_id: str) -> dict:
    exp_dir = EXP_ROOT / experiment_id
    if not exp_dir.exists():
        # try without suffix
        # find closest
        candidates = list(EXP_ROOT.glob(f"{experiment_id}*"))
        if not candidates:
            raise FileNotFoundError(f"experiment {experiment_id} not found")
        exp_dir = candidates[0]
    cfg = json.loads((exp_dir / "config.json").read_text())
    return run_experiment(ticker=cfg["ticker"], horizon=cfg["horizon"], start=cfg["start"], end=cfg["end"], seed=cfg["seed"])


def main():
    p = argparse.ArgumentParser(description="MVP Research loop — single command")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reproduce", type=str, default=None, help="experiment_id to reproduce")
    args = p.parse_args()
    if args.reproduce:
        report = reproduce(args.reproduce)
    else:
        report = run_experiment(args.ticker, args.horizon, args.start, args.end, args.seed)
    print(json.dumps({"experiment_id": report["experiment_id"], "decision": report["decision"], "delta_acc": report["improvement_delta_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
