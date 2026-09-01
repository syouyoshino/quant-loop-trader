"""Leakage-safe random-start replay robustness testing.

Random replay can operate as a standalone BTC diagnostic or replay one verified
experiment candidate. Candidate mode reconstructs the candidate's model type,
feature columns, model parameters, horizon, seed, and sealed dataset snapshot.
Unsupported feature sets fail closed rather than silently replaying a different
strategy. Campaign holdout observations are never loaded or sampled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from quant_loop_trader.core import significance
from quant_loop_trader.data import coverage_check, dataset_metadata, fetch_ohlcv, gap_check
from quant_loop_trader.evaluation import autopsy, evaluate
from quant_loop_trader.experiment import make_labels
from quant_loop_trader.features import add_improved_features, improved_feature_columns
from quant_loop_trader.market import campaign_holdout_start, campaign_id
from quant_loop_trader.models.registry import build_model

DEFAULT_RUNS = 100
DEFAULT_TRADE_DAYS = 180
DEFAULT_MIN_TRAINING_DAYS = 730
_REQUIRED_OHLCV = (
    "event_time", "available_time", "open", "high", "low", "close", "volume",
)


@dataclass(frozen=True)
class CandidateReplaySpec:
    experiment_id: str
    ticker: str
    horizon: int
    model_seed: int
    model_type: str
    model_params: dict
    feature_columns: tuple[str, ...]
    dataset_snapshot: Path
    dataset_id: str | None
    dataset_checksum: str | None
    spec_fingerprint: str | None
    model_version: str | None
    feature_version: str | None
    dataset_snapshot_sha256: str | None


def _as_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_experiment_id(experiment_id: str) -> str:
    if not experiment_id or Path(experiment_id).name != experiment_id:
        raise ValueError("invalid_candidate_experiment_id")
    return experiment_id


def _load_candidate_spec(experiment_id: str, root: Path) -> CandidateReplaySpec:
    """Open one verified candidate bundle and derive its replayable strategy spec."""
    from quant_loop_trader.bundle import ExperimentBundle

    experiment_id = _safe_experiment_id(experiment_id)
    bundle = ExperimentBundle.open_verified(experiment_id, root / "data" / "experiments")
    cfg = bundle.config
    report = bundle.report
    ticker = str(cfg["ticker"]).upper()

    candidate_boundary = cfg.get("campaign_holdout_start")
    active_boundary = campaign_holdout_start(ticker)
    if candidate_boundary != active_boundary:
        raise ValueError(
            "candidate_campaign_holdout_mismatch:"
            f"candidate={candidate_boundary}:active={active_boundary}"
        )

    feature_cols = cfg.get("feature_columns")
    if not feature_cols:
        feature_cols = report.get("parameters", {}).get("feature_columns")
    if not feature_cols:
        # Backward compatibility for current sealed bundles, whose improved
        # feature identity is versioned but whose column list predates this field.
        feature_cols = improved_feature_columns()

    model_type = (
        cfg.get("model_type")
        or report.get("parameters", {}).get("model_type")
        or "logistic"
    )
    model_params = cfg.get("model_params") or report.get("parameters", {}).get("model_params") or {}
    snapshot = Path(cfg.get("dataset_snapshot") or bundle.dataset_snapshot)
    if not snapshot.exists():
        raise FileNotFoundError(f"candidate_dataset_snapshot_missing:{snapshot}")

    return CandidateReplaySpec(
        experiment_id=experiment_id,
        ticker=ticker,
        horizon=int(cfg["horizon"]),
        model_seed=int(cfg["seed"]),
        model_type=str(model_type),
        model_params=dict(model_params),
        feature_columns=tuple(str(c) for c in feature_cols),
        dataset_snapshot=snapshot,
        dataset_id=cfg.get("dataset_id"),
        dataset_checksum=cfg.get("dataset_checksum"),
        spec_fingerprint=cfg.get("spec_fingerprint"),
        model_version=cfg.get("model_version"),
        feature_version=cfg.get("feature_version_improved"),
        dataset_snapshot_sha256=bundle.lock.get("dataset_snapshot_sha256"),
    )


def _resolve_data_end(ticker: str, data_end: str | None) -> tuple[date, date | None]:
    """Resolve the last bar random replay may load, always before fixed holdout."""
    holdout_raw = campaign_holdout_start(ticker)
    holdout = _as_date(holdout_raw) if holdout_raw else None
    if data_end is None:
        if holdout is None:
            raise ValueError("data_end_required_without_fixed_holdout")
        resolved = holdout - timedelta(days=1)
    else:
        resolved = _as_date(data_end)
    if holdout is not None and resolved >= holdout:
        raise ValueError(
            f"random_replay_data_end_reaches_holdout:data_end={resolved}:holdout={holdout}"
        )
    return resolved, holdout


def _validate_source_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Require the complete PIT OHLCV contract before a replay snapshot is sealed."""
    missing = [c for c in _REQUIRED_OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"source_snapshot_missing_columns:{','.join(missing)}")
    try:
        df = df.with_columns(
            pl.col("event_time").cast(pl.Date),
            pl.col("available_time").cast(pl.Date),
            *[pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close", "volume")],
        ).sort("event_time")
    except Exception as exc:
        raise ValueError(f"source_snapshot_schema_invalid:{str(exc)[:120]}") from exc
    if any(df[c].null_count() for c in _REQUIRED_OHLCV):
        raise ValueError("source_snapshot_required_column_nulls")
    if df.filter(pl.col("available_time") < pl.col("event_time")).height:
        raise ValueError("source_snapshot_pit_violation")
    if df["event_time"].n_unique() != df.height:
        raise ValueError("source_snapshot_duplicate_event_time")
    return df


def _load_research_snapshot(
    ticker: str,
    data_start: date,
    data_end: date,
    root: Path,
    source_snapshot: str | Path | None = None,
) -> tuple[pl.DataFrame, dict, Path]:
    """Acquire once, validate once, seal once, then reuse one immutable frame."""
    ticker = ticker.upper()
    if source_snapshot is None:
        df, source = fetch_ohlcv(ticker, data_start.isoformat(), data_end.isoformat())
    else:
        source_path = Path(source_snapshot)
        if not source_path.exists():
            raise FileNotFoundError(f"source snapshot not found: {source_path}")
        df = pl.read_parquet(str(source_path))
        source = "snapshot_random_replay"

    df = _validate_source_frame(df)
    df = df.filter(
        (pl.col("event_time") >= pl.lit(data_start))
        & (pl.col("event_time") <= pl.lit(data_end))
    ).sort("event_time")
    gap_check(df, ticker=ticker)
    coverage_check(df, ticker=ticker, start=data_start.isoformat(), end=data_end.isoformat())
    if df.height == 0:
        raise ValueError("random_replay_empty_dataset")

    meta = dataset_metadata(
        df,
        ticker,
        source,
        extra_provenance={
            "purpose": "random_start_replay",
            "data_start": data_start.isoformat(),
            "data_end": data_end.isoformat(),
        },
    )
    snap_dir = root / "data" / "datasets"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"{meta['dataset_id']}.parquet"
    if not snap_path.exists():
        df.write_parquet(str(snap_path))
    return df, meta, snap_path


def prepare_replay_frame(
    df: pl.DataFrame,
    horizon: int,
    feature_cols: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Create PIT features and labels once from the sealed research snapshot."""
    h = max(1, int(horizon))
    cols = list(feature_cols or improved_feature_columns())
    supported = set(improved_feature_columns())
    unsupported = [c for c in cols if c not in supported]
    if unsupported:
        raise ValueError(f"unsupported_candidate_features:{','.join(unsupported)}")
    frame = add_improved_features(make_labels(df.sort("event_time"), h))
    frame = frame.drop_nulls(subset=cols + ["label"]).sort("event_time")
    if frame.height == 0:
        raise ValueError("random_replay_no_usable_rows")
    return frame


def eligible_start_indices(
    frame: pl.DataFrame,
    *,
    horizon: int,
    trade_days: int,
    min_training_days: int,
    sample_start: date,
    sample_end: date,
) -> list[int]:
    """Return replay starts with enough purged history and forward observations."""
    h = max(1, int(horizon))
    trade_days = int(trade_days)
    min_training_days = int(min_training_days)
    if trade_days <= h:
        raise ValueError("trade_days_must_exceed_horizon")
    if min_training_days < 50:
        raise ValueError("min_training_days_too_small")
    if sample_end < sample_start:
        raise ValueError("sample_end_before_sample_start")
    dates = frame["event_time"].to_list()
    first_idx = min_training_days + h
    last_idx = frame.height - trade_days
    if last_idx < first_idx:
        return []
    return [
        idx for idx in range(first_idx, last_idx + 1)
        if sample_start <= dates[idx] <= sample_end
    ]


def select_balanced_starts(
    frame: pl.DataFrame,
    candidates: list[int],
    *,
    runs: int,
    seed: int,
    min_start_gap_days: int = 0,
) -> list[int]:
    """Sample unique starts roughly evenly across calendar years."""
    runs = int(runs)
    min_start_gap_days = max(0, int(min_start_gap_days))
    if runs <= 0:
        raise ValueError("runs_must_be_positive")
    if not candidates:
        raise ValueError("no_eligible_random_replay_starts")
    dates = frame["event_time"].to_list()
    by_year: dict[int, list[int]] = {}
    for idx in candidates:
        by_year.setdefault(dates[idx].year, []).append(idx)
    rng = random.Random(int(seed))
    years = sorted(by_year)
    rng.shuffle(years)
    for bucket in by_year.values():
        rng.shuffle(bucket)
    selected: list[int] = []
    selected_dates: list[date] = []

    def acceptable(idx: int) -> bool:
        if min_start_gap_days == 0:
            return True
        d = dates[idx]
        return all(abs((d - other).days) >= min_start_gap_days for other in selected_dates)

    while len(selected) < runs:
        progress = False
        for year in years:
            bucket = by_year[year]
            while bucket:
                idx = bucket.pop()
                if not acceptable(idx):
                    continue
                selected.append(idx)
                selected_dates.append(dates[idx])
                progress = True
                break
            if len(selected) >= runs:
                break
        if not progress:
            break
    if len(selected) < runs:
        raise ValueError(
            "not_enough_distinct_replay_starts:"
            f"requested={runs}:available_after_spacing={len(selected)}:"
            f"min_start_gap_days={min_start_gap_days}"
        )
    return selected


def build_replay_window(
    frame: pl.DataFrame,
    start_idx: int,
    *,
    horizon: int,
    trade_days: int,
    min_training_days: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build one expanding-history train window and one fixed forward test window."""
    h = max(1, int(horizon))
    train_end = int(start_idx) - h
    if train_end < int(min_training_days):
        raise ValueError("insufficient_purged_training_history")
    train = frame.slice(0, train_end)
    test = frame.slice(int(start_idx), int(trade_days))
    if test.height != int(trade_days):
        raise ValueError("insufficient_forward_replay_window")
    if train.height == 0 or test.height == 0:
        raise ValueError("empty_random_replay_window")
    if train["event_time"].max() >= test["event_time"].min():
        raise AssertionError("random_replay_time_leakage")
    return train, test


def _train_evaluate_candidate(
    train: pl.DataFrame,
    test: pl.DataFrame,
    feat_cols: list[str],
    horizon: int,
    *,
    model_type: str,
    model_seed: int,
    model_params: dict,
    ticker: str,
) -> dict:
    """Train exactly the requested registry model on one replay window."""
    X_train = train.select(feat_cols).to_numpy()
    y_train = train["label"].to_numpy()
    X_test = test.select(feat_cols).to_numpy()
    y_test = test["label"].to_numpy()
    prices_test = test["close"].to_numpy()
    model = build_model(model_type, seed=int(model_seed), **dict(model_params))
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
    metrics = evaluate(y_test, y_pred, y_prob, prices_test, horizon, ticker=ticker)
    metrics["stat_pvalue"] = significance(y_test, y_pred, horizon=horizon).pvalue
    return {"metrics": metrics, "error_analysis": autopsy(test, y_test, y_pred), "model": model}


def _overlap_diagnostics(rows: list[dict]) -> dict:
    windows = sorted(
        ((_as_date(row["start_date"]), _as_date(row["trade_end"])) for row in rows),
        key=lambda item: item[0],
    )
    if len(windows) < 2:
        return {
            "median_start_gap_days": None,
            "overlapping_window_fraction": 0.0,
            "statistical_independence": "not_assumed",
        }
    gaps = [(windows[i][0] - windows[i - 1][0]).days for i in range(1, len(windows))]
    overlaps = 0
    furthest_end = windows[0][1]
    for start, end in windows[1:]:
        if start <= furthest_end:
            overlaps += 1
        if end > furthest_end:
            furthest_end = end
    return {
        "median_start_gap_days": float(np.median(gaps)),
        "overlapping_window_fraction": float(overlaps / (len(windows) - 1)),
        "statistical_independence": "not_assumed",
    }


def summarize_replays(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot_summarize_empty_replays")
    strategy = np.asarray([row["strategy_return"] for row in rows], dtype=float)
    benchmark = np.asarray([row["benchmark_return"] for row in rows], dtype=float)
    sharpe = np.asarray([row["sharpe"] for row in rows], dtype=float)
    drawdown = np.asarray([row["max_drawdown"] for row in rows], dtype=float)
    cost_25 = np.asarray([row["return_25bps_worst_phase"] for row in rows], dtype=float)
    year_counts = Counter(str(_as_date(row["start_date"]).year) for row in rows)
    out = {
        "runs": len(rows),
        "unique_start_dates": len({row["start_date"] for row in rows}),
        "profitable_fraction": float(np.mean(strategy > 0)),
        "beat_benchmark_fraction": float(np.mean(strategy > benchmark)),
        "positive_25bps_fraction": float(np.mean(cost_25 > 0)),
        "median_return": float(np.median(strategy)),
        "median_benchmark_return": float(np.median(benchmark)),
        "median_excess_return": float(np.median(strategy - benchmark)),
        "median_sharpe": float(np.median(sharpe)),
        "median_max_drawdown": float(np.median(drawdown)),
        "p10_return": float(np.quantile(strategy, 0.10)),
        "worst_return": float(np.min(strategy)),
        "best_return": float(np.max(strategy)),
        "starts_by_year": dict(sorted(year_counts.items())),
    }
    out.update(_overlap_diagnostics(rows))
    return out


def _candidate_evidence_file(root: Path, experiment_id: str, replay_id: str) -> Path:
    experiment_id = _safe_experiment_id(experiment_id)
    d = root / "data" / "random_replays" / "by_experiment" / experiment_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{replay_id}.json"


def latest_replay_for_experiment(experiment_id: str, root: str | Path | None = None) -> dict | None:
    """Return the latest externally-bound replay diagnostic for one experiment."""
    root_path = Path(root) if root is not None else Path(os.environ.get("QLT_ROOT", Path.cwd()))
    experiment_id = _safe_experiment_id(experiment_id)
    d = root_path / "data" / "random_replays" / "by_experiment" / experiment_id
    if not d.exists():
        return None
    rows = []
    for p in d.glob("*.json"):
        try:
            row = json.loads(p.read_text())
            summary_path = Path(row["summary_json"])
            if not summary_path.is_absolute():
                summary_path = root_path / summary_path
            expected = row.get("summary_sha256")
            if expected and (not summary_path.exists() or _sha_file(summary_path) != expected):
                continue
            rows.append(row)
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return max(rows, key=lambda r: r.get("created_at", "")) if rows else None


def run_random_replay(
    *,
    experiment_id: str | None = None,
    ticker: str | None = None,
    horizon: int | None = None,
    runs: int = DEFAULT_RUNS,
    trade_days: int = DEFAULT_TRADE_DAYS,
    min_training_days: int = DEFAULT_MIN_TRAINING_DAYS,
    data_start: str | None = None,
    data_end: str | None = None,
    sample_start: str | None = None,
    sample_end: str | None = None,
    seed: int | None = None,
    min_start_gap_days: int = 0,
    source_snapshot: str | Path | None = None,
    root: str | Path | None = None,
) -> dict:
    """Run repeated random-start historical replays and persist compact evidence."""
    root_path = Path(root) if root is not None else Path(os.environ.get("QLT_ROOT", Path.cwd()))
    candidate = _load_candidate_spec(experiment_id, root_path) if experiment_id else None

    if candidate:
        if ticker is not None and ticker.upper() != candidate.ticker:
            raise ValueError("candidate_ticker_override_forbidden")
        if horizon is not None and int(horizon) != candidate.horizon:
            raise ValueError("candidate_horizon_override_forbidden")
        if source_snapshot is not None:
            raise ValueError("candidate_source_snapshot_override_forbidden")
        ticker = candidate.ticker
        h = candidate.horizon
        model_seed = candidate.model_seed
        model_type = candidate.model_type
        model_params = candidate.model_params
        feat_cols = list(candidate.feature_columns)
        source_snapshot = candidate.dataset_snapshot
        resolved_data_start = _as_date(data_start or str(candidate.dataset_snapshot and candidate.dataset_snapshot)) if False else _as_date(data_start or "2018-01-01")
        # Existing bundles record their requested research start. Prefer that over
        # the generic default when candidate mode is used.
        from quant_loop_trader.bundle import ExperimentBundle
        candidate_cfg = ExperimentBundle.open_verified(
            candidate.experiment_id, root_path / "data" / "experiments"
        ).config
        resolved_data_start = _as_date(data_start or candidate_cfg.get("start", "2018-01-01"))
        sampling_seed = int(seed if seed is not None else 42)
    else:
        ticker = (ticker or "BTCUSD").upper()
        h = max(1, int(horizon if horizon is not None else 5))
        model_seed = int(seed if seed is not None else 42)
        model_type = "logistic"
        model_params = {}
        feat_cols = improved_feature_columns()
        resolved_data_start = _as_date(data_start or "2018-01-01")
        sampling_seed = int(seed if seed is not None else 42)

    resolved_data_end, holdout = _resolve_data_end(ticker, data_end)
    if resolved_data_end <= resolved_data_start:
        raise ValueError("data_end_must_follow_data_start")
    resolved_sample_start = _as_date(sample_start or resolved_data_start)
    resolved_sample_end = _as_date(sample_end) if sample_end else resolved_data_end
    if resolved_sample_start < resolved_data_start:
        raise ValueError("sample_start_before_data_start")
    if resolved_sample_end > resolved_data_end:
        raise ValueError("sample_end_after_data_end")

    df, meta, snap_path = _load_research_snapshot(
        ticker,
        resolved_data_start,
        resolved_data_end,
        root_path,
        source_snapshot=source_snapshot,
    )
    frame = prepare_replay_frame(df, h, feat_cols)
    candidates = eligible_start_indices(
        frame,
        horizon=h,
        trade_days=trade_days,
        min_training_days=min_training_days,
        sample_start=resolved_sample_start,
        sample_end=resolved_sample_end,
    )
    starts = select_balanced_starts(
        frame,
        candidates,
        runs=runs,
        seed=sampling_seed,
        min_start_gap_days=min_start_gap_days,
    )

    rows: list[dict] = []
    for replay_no, start_idx in enumerate(starts, start=1):
        train, test = build_replay_window(
            frame,
            start_idx,
            horizon=h,
            trade_days=trade_days,
            min_training_days=min_training_days,
        )
        result = _train_evaluate_candidate(
            train,
            test,
            feat_cols,
            h,
            model_type=model_type,
            model_seed=model_seed,
            model_params=model_params,
            ticker=ticker,
        )
        metrics = result["metrics"]
        strategy_return = float(metrics["cumulative_return_strategy_liquidated"])
        benchmark_return = float(metrics["cumulative_return_benchmark"])
        rows.append({
            "replay": replay_no,
            "start_date": str(test["event_time"].min()),
            "trade_end": str(test["event_time"].max()),
            "train_start": str(train["event_time"].min()),
            "train_end": str(train["event_time"].max()),
            "train_rows": train.height,
            "trade_rows": test.height,
            "strategy_return": strategy_return,
            "benchmark_return": benchmark_return,
            "excess_return": strategy_return - benchmark_return,
            "sharpe": float(metrics.get("sharpe_strategy_liquidated", metrics["sharpe_strategy"])),
            "max_drawdown": float(metrics["max_drawdown_strategy"]),
            "accuracy": float(metrics["accuracy"]),
            "return_25bps_worst_phase": float(
                metrics.get("cost_sensitivity_compounded", {}).get("25", 0.0)
            ),
            "profitable": strategy_return > 0,
            "beat_benchmark": strategy_return > benchmark_return,
        })

    summary = summarize_replays(rows)
    config = {
        "candidate_experiment_id": candidate.experiment_id if candidate else None,
        "candidate_spec_fingerprint": candidate.spec_fingerprint if candidate else None,
        "candidate_dataset_id": candidate.dataset_id if candidate else None,
        "candidate_dataset_checksum": candidate.dataset_checksum if candidate else None,
        "candidate_dataset_snapshot_sha256": (
            candidate.dataset_snapshot_sha256 if candidate else None
        ),
        "candidate_model_version": candidate.model_version if candidate else None,
        "candidate_feature_version": candidate.feature_version if candidate else None,
        "ticker": ticker,
        "horizon": h,
        "runs": int(runs),
        "trade_days": int(trade_days),
        "min_training_days": int(min_training_days),
        "data_start": resolved_data_start.isoformat(),
        "data_end": resolved_data_end.isoformat(),
        "sample_start": resolved_sample_start.isoformat(),
        "sample_end": resolved_sample_end.isoformat(),
        "sampling_seed": sampling_seed,
        "model_seed": model_seed,
        "model_type": model_type,
        "model_params": model_params,
        "min_start_gap_days": int(min_start_gap_days),
        "campaign_id": campaign_id(ticker),
        "campaign_holdout_start": holdout.isoformat() if holdout else None,
        "dataset_id": meta["dataset_id"],
        "dataset_checksum": meta["checksum"],
        "dataset_snapshot": str(snap_path),
        "feature_set": feat_cols,
        "sampling_policy": "seeded_unique_round_robin_by_calendar_year",
        "training_policy": "expanding_history_with_horizon_purge",
        "independence_policy": "overlap_reported_not_assumed_independent",
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    replay_id = f"{day}_{ticker}_{h}d_rr_{fingerprint}"
    replay_root = root_path / "data" / "random_replays"
    replay_root.mkdir(parents=True, exist_ok=True)
    replay_dir = replay_root / replay_id
    suffix = 0
    while replay_dir.exists():
        suffix += 1
        replay_dir = replay_root / f"{replay_id}_r{suffix}"
    replay_dir.mkdir(parents=True, exist_ok=False)

    evidence_path = (
        _candidate_evidence_file(root_path, candidate.experiment_id, replay_dir.name)
        if candidate else None
    )
    report = {
        "random_replay_id": replay_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "summary": summary,
        "artifacts": {
            "runs_csv": str(replay_dir / "runs.csv"),
            "summary_json": str(replay_dir / "summary.json"),
            "config_json": str(replay_dir / "config.json"),
            "candidate_evidence_json": str(evidence_path) if evidence_path else None,
        },
    }
    pl.DataFrame(rows).write_csv(str(replay_dir / "runs.csv"))
    (replay_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    summary_path = replay_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    if evidence_path is not None:
        evidence_path.write_text(json.dumps({
            "candidate_experiment_id": candidate.experiment_id,
            "random_replay_id": replay_dir.name,
            "created_at": report["created_at"],
            "summary": summary,
            "summary_json": str(summary_path),
            "summary_sha256": _sha_file(summary_path),
        }, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe random-start robustness replay for Quant Loop Trader"
    )
    parser.add_argument("--experiment", default=None, help="verified candidate experiment id")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--trade-days", type=int, default=DEFAULT_TRADE_DAYS)
    parser.add_argument("--min-training-days", type=int, default=DEFAULT_MIN_TRAINING_DAYS)
    parser.add_argument("--data-start", default=None)
    parser.add_argument("--data-end", default=None)
    parser.add_argument("--sample-start", default=None)
    parser.add_argument("--sample-end", default=None)
    parser.add_argument("--seed", type=int, default=None, help="sampling seed; standalone model seed too")
    parser.add_argument("--min-start-gap-days", type=int, default=0)
    parser.add_argument("--source-snapshot", default=None)
    args = parser.parse_args()
    report = run_random_replay(
        experiment_id=args.experiment,
        ticker=args.ticker,
        horizon=args.horizon,
        runs=args.runs,
        trade_days=args.trade_days,
        min_training_days=args.min_training_days,
        data_start=args.data_start,
        data_end=args.data_end,
        sample_start=args.sample_start,
        sample_end=args.sample_end,
        seed=args.seed,
        min_start_gap_days=args.min_start_gap_days,
        source_snapshot=args.source_snapshot,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
