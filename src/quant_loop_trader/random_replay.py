"""Leakage-safe random-start replay robustness testing.

This module repeatedly drops the current candidate strategy into randomly sampled
historical start dates. Each replay trains on information available strictly before
the sampled start, purges the prediction horizon at the boundary, and evaluates a
fixed forward trading window. The campaign holdout is never sampled or loaded.

Random replay is a robustness diagnostic, not a replacement for the final untouched
holdout and not a claim that overlapping replay windows are statistically independent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from quant_loop_trader.data import (
    coverage_check,
    dataset_metadata,
    fetch_ohlcv,
    gap_check,
)
from quant_loop_trader.experiment import make_labels, train_evaluate_from
from quant_loop_trader.features import add_improved_features, improved_feature_columns
from quant_loop_trader.market import campaign_holdout_start, campaign_id

DEFAULT_RUNS = 100
DEFAULT_TRADE_DAYS = 180
DEFAULT_MIN_TRAINING_DAYS = 730


def _as_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


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


def _load_research_snapshot(
    ticker: str,
    data_start: date,
    data_end: date,
    root: Path,
    source_snapshot: str | Path | None = None,
) -> tuple[pl.DataFrame, dict, Path]:
    """Acquire once, seal once, then reuse the same immutable frame for all replays."""
    ticker = ticker.upper()
    if source_snapshot is None:
        df, source = fetch_ohlcv(ticker, data_start.isoformat(), data_end.isoformat())
    else:
        source_path = Path(source_snapshot)
        if not source_path.exists():
            raise FileNotFoundError(f"source snapshot not found: {source_path}")
        df = pl.read_parquet(str(source_path))
        if "event_time" not in df.columns:
            raise ValueError("source_snapshot_missing_event_time")
        df = df.with_columns(pl.col("event_time").cast(pl.Date))
        if "available_time" in df.columns:
            df = df.with_columns(pl.col("available_time").cast(pl.Date))
        df = df.filter(
            (pl.col("event_time") >= pl.lit(data_start))
            & (pl.col("event_time") <= pl.lit(data_end))
        ).sort("event_time")
        source = "snapshot_random_replay"
        gap_check(df, ticker=ticker)
        coverage_check(
            df,
            ticker=ticker,
            start=data_start.isoformat(),
            end=data_end.isoformat(),
        )

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


def prepare_replay_frame(df: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Create PIT features and labels once from the sealed research snapshot."""
    h = max(1, int(horizon))
    feat_cols = improved_feature_columns()
    frame = add_improved_features(make_labels(df.sort("event_time"), h))
    frame = frame.drop_nulls(subset=feat_cols + ["label"]).sort("event_time")
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
        idx
        for idx in range(first_idx, last_idx + 1)
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
    """Sample unique start dates roughly evenly across calendar years.

    The round-robin sampler avoids accidentally filling most replays from one BTC
    regime/year. Optional spacing can reduce near-duplicate starts, but overlapping
    forward windows are still allowed and are explicitly reported as correlated.
    """
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
    """Build one expanding-history train window and one fixed forward test window.

    The last ``horizon`` observations before the replay start are removed from
    training. Their labels depend on prices at/after the replay start and would
    otherwise leak the simulated future into model fitting.
    """
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


def run_random_replay(
    *,
    ticker: str = "BTCUSD",
    horizon: int = 5,
    runs: int = DEFAULT_RUNS,
    trade_days: int = DEFAULT_TRADE_DAYS,
    min_training_days: int = DEFAULT_MIN_TRAINING_DAYS,
    data_start: str = "2018-01-01",
    data_end: str | None = None,
    sample_start: str | None = None,
    sample_end: str | None = None,
    seed: int = 42,
    min_start_gap_days: int = 0,
    source_snapshot: str | Path | None = None,
    root: str | Path | None = None,
) -> dict:
    """Run repeated random-start historical replays and persist a compact report."""
    ticker = ticker.upper()
    h = max(1, int(horizon))
    resolved_data_start = _as_date(data_start)
    resolved_data_end, holdout = _resolve_data_end(ticker, data_end)
    if resolved_data_end <= resolved_data_start:
        raise ValueError("data_end_must_follow_data_start")

    resolved_sample_start = _as_date(sample_start or data_start)
    resolved_sample_end = _as_date(sample_end) if sample_end else resolved_data_end
    if resolved_sample_start < resolved_data_start:
        raise ValueError("sample_start_before_data_start")
    if resolved_sample_end > resolved_data_end:
        raise ValueError("sample_end_after_data_end")

    root_path = Path(root) if root is not None else Path(os.environ.get("QLT_ROOT", Path.cwd()))
    df, meta, snap_path = _load_research_snapshot(
        ticker,
        resolved_data_start,
        resolved_data_end,
        root_path,
        source_snapshot=source_snapshot,
    )
    frame = prepare_replay_frame(df, h)
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
        seed=seed,
        min_start_gap_days=min_start_gap_days,
    )

    feat_cols = improved_feature_columns()
    rows: list[dict] = []
    for replay_no, start_idx in enumerate(starts, start=1):
        train, test = build_replay_window(
            frame,
            start_idx,
            horizon=h,
            trade_days=trade_days,
            min_training_days=min_training_days,
        )
        result = train_evaluate_from(
            train,
            test,
            feat_cols,
            h,
            seed=int(seed),
            ticker=ticker,
        )
        metrics = result["metrics"]
        strategy_return = float(metrics["cumulative_return_strategy_liquidated"])
        benchmark_return = float(metrics["cumulative_return_benchmark"])
        row = {
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
            "sharpe": float(
                metrics.get("sharpe_strategy_liquidated", metrics["sharpe_strategy"])
            ),
            "max_drawdown": float(metrics["max_drawdown_strategy"]),
            "accuracy": float(metrics["accuracy"]),
            "return_25bps_worst_phase": float(
                metrics.get("cost_sensitivity_compounded", {}).get("25", 0.0)
            ),
            "profitable": strategy_return > 0,
            "beat_benchmark": strategy_return > benchmark_return,
        }
        rows.append(row)

    summary = summarize_replays(rows)
    config = {
        "ticker": ticker,
        "horizon": h,
        "runs": int(runs),
        "trade_days": int(trade_days),
        "min_training_days": int(min_training_days),
        "data_start": resolved_data_start.isoformat(),
        "data_end": resolved_data_end.isoformat(),
        "sample_start": resolved_sample_start.isoformat(),
        "sample_end": resolved_sample_end.isoformat(),
        "seed": int(seed),
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

    report = {
        "random_replay_id": replay_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "summary": summary,
        "artifacts": {
            "runs_csv": str(replay_dir / "runs.csv"),
            "summary_json": str(replay_dir / "summary.json"),
            "config_json": str(replay_dir / "config.json"),
        },
    }
    pl.DataFrame(rows).write_csv(str(replay_dir / "runs.csv"))
    (replay_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    (replay_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe random-start robustness replay for Quant Loop Trader"
    )
    parser.add_argument("--ticker", default="BTCUSD")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--trade-days", type=int, default=DEFAULT_TRADE_DAYS)
    parser.add_argument("--min-training-days", type=int, default=DEFAULT_MIN_TRAINING_DAYS)
    parser.add_argument("--data-start", default="2018-01-01")
    parser.add_argument("--data-end", default=None)
    parser.add_argument("--sample-start", default=None)
    parser.add_argument("--sample-end", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-start-gap-days", type=int, default=0)
    parser.add_argument("--source-snapshot", default=None)
    args = parser.parse_args()

    report = run_random_replay(
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
