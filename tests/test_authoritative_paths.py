"""Regression tests for the post-simplification authoritative path."""
import datetime
import json

import duckdb
import polars as pl
import pytest


def test_replay_accepts_content_addressed_snapshot_name(tmp_path):
    from quant_loop_trader.replay import ReplayEngine

    dates = [datetime.date(2024, 1, d) for d in (2, 3, 4)]
    df = pl.DataFrame({
        "event_time": dates,
        "available_time": dates,
        "close": [100.0, 101.0, 102.0],
    })
    path = tmp_path / "SPY_2024-01-02_2024-01-04_deadbeef.parquet"
    df.write_parquet(path)

    snap = ReplayEngine(path, ticker="SPY").get_snapshot("SPY", "2024-01-03")
    assert snap.height == 2
    assert snap["event_time"].max() == datetime.date(2024, 1, 3)


def test_new_experiment_seals_content_addressed_snapshot(isolated_research):
    from quant_loop_trader import data as dm
    from quant_loop_trader.experiment import run_experiment

    report = run_experiment(
        ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=7001
    )
    exp_dir = isolated_research / report["experiment_id"]
    config = json.loads((exp_dir / "config.json").read_text())
    lock = json.loads((exp_dir / "predictions.lock").read_text())
    snapshot = dm.PROC_DIR.parent / "datasets" / f"{config['dataset_id']}.parquet"

    assert snapshot.exists()
    assert config["dataset_snapshot"] == str(snapshot)
    assert report["data_dependencies"] == [str(snapshot)]
    assert "dataset_snapshot_sha256" in lock
    assert "dataset_parquet" not in lock


def test_mutable_cache_drift_cannot_change_verified_experiment(isolated_research):
    from quant_loop_trader import data as dm
    from quant_loop_trader.agents import independent_replication
    from quant_loop_trader.bundle import ExperimentBundle
    from quant_loop_trader.experiment import EXP_ROOT, run_experiment

    report = run_experiment(
        ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=7002
    )

    # Change the acquisition cache after sealing. Authoritative work must ignore it.
    cache = dm.PROC_DIR / "SPY.parquet"
    cached = pl.read_parquet(cache)
    cached = cached.with_columns((pl.col("close") * 1.25).alias("close"))
    cached.write_parquet(cache)

    bundle = ExperimentBundle.open_verified(report["experiment_id"], EXP_ROOT)
    assert bundle.dataset_snapshot.exists()
    replication = independent_replication(report["experiment_id"])
    assert not any("replication_mismatch" in issue for issue in replication["issues_found"])
    assert not any("replication_metric_mismatch" in issue for issue in replication["issues_found"])


def test_snapshot_tampering_fails_closed(isolated_research):
    from quant_loop_trader import data as dm
    from quant_loop_trader.bundle import BundleIntegrityError, ExperimentBundle
    from quant_loop_trader.experiment import EXP_ROOT, run_experiment

    report = run_experiment(
        ticker="SPY", horizon=5, start="2020-01-01", end="2024-12-31", seed=7003
    )
    dataset_id = report["config"]["dataset_id"]
    snapshot = dm.PROC_DIR.parent / "datasets" / f"{dataset_id}.parquet"
    df = pl.read_parquet(snapshot)
    df = df.with_columns(
        pl.when(pl.arange(0, df.height) == 0)
        .then(pl.col("close") + 1.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    df.write_parquet(snapshot)

    with pytest.raises(BundleIntegrityError, match="dataset_snapshot"):
        ExperimentBundle.open_verified(report["experiment_id"], EXP_ROOT)


def test_candidate_pvalue_uses_canonical_significance(isolated_research):
    from quant_loop_trader.core import significance
    from quant_loop_trader.experiment import run_experiment

    report = run_experiment(
        ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=7004
    )
    pred = pl.read_parquet(
        isolated_research / report["experiment_id"] / "predictions_improved.parquet"
    )
    expected = significance(
        pred["y_true"].to_numpy(),
        pred["y_pred"].to_numpy(),
        horizon=report["config"]["horizon"],
    ).pvalue
    assert report["candidate_stat_pvalue"] == pytest.approx(expected, abs=1e-15)


def test_reproduction_persists_lineage_without_rewriting_dataset_provenance(isolated_research):
    from quant_loop_trader import data as dm
    from quant_loop_trader.bundle import ExperimentBundle
    from quant_loop_trader.experiment import EXP_ROOT, reproduce, run_experiment

    original = run_experiment(
        ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=7005
    )
    dataset_id = original["config"]["dataset_id"]
    con = duckdb.connect(str(dm.DB_PATH))
    source_before = con.execute(
        "SELECT source FROM datasets WHERE dataset_id=?", [dataset_id]
    ).fetchone()[0]
    con.close()

    child = reproduce(original["experiment_id"])
    child_dir = isolated_research / child["experiment_id"]
    persisted = json.loads((child_dir / "report.json").read_text())
    assert persisted["parent_experiment_id"] == original["experiment_id"]
    assert persisted["reproduction_check"]["reproduced"] is True
    ExperimentBundle.open_verified(child["experiment_id"], EXP_ROOT)

    con = duckdb.connect(str(dm.DB_PATH))
    source_after = con.execute(
        "SELECT source FROM datasets WHERE dataset_id=?", [dataset_id]
    ).fetchone()[0]
    con.close()
    assert source_after == source_before


def test_invalid_lifecycle_evidence_cannot_fall_through_to_champion():
    from quant_loop_trader.core import LifecycleEvidence, final_state

    with pytest.raises(ValueError, match="invalid research_screen"):
        final_state(LifecycleEvidence("UNKNOWN", validation="PASS", holdout="PASS"))
    with pytest.raises(ValueError, match="requires validation=PASS"):
        final_state(LifecycleEvidence("KEEP", validation="NOT_RUN", holdout="PASS"))
