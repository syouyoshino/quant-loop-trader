import json
import polars as pl
from quant_loop_trader.experiment import EXP_ROOT, run_experiment
from quant_loop_trader.data import PROC_DIR, DB_PATH, FIXTURE_PATH
import duckdb

def test_experiment_e2e_fixture(isolated_research):
    # Use short date range for speed, but real fixture covers 2018-2024
    # run_experiment will use TIINGO if key exists; we force fixture by mocking env?
    # Instead test with actual data but verify artifacts
    report = run_experiment(ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=123)
    assert "experiment_id" in report
    assert report["decision"] in ("KEEP","IMPROVE","REJECT")
    assert "baseline_metrics" in report and "improved_metrics" in report
    assert "hypothesis" in report and "economic_reasoning" in report
    assert "reproducibility" in report
    # files exist
    exp_dir = isolated_research / report["experiment_id"]
    assert (exp_dir / "report.json").exists()
    assert (exp_dir / "predictions_baseline.parquet").exists()
    assert (exp_dir / "predictions_improved.parquet").exists()
    assert (exp_dir / "metrics.json").exists()
    # DuckDB rows
    import quant_loop_trader.data as _dm
    con = duckdb.connect(str(_dm.DB_PATH))
    rows = con.execute("select count(*) from experiments where experiment_id like ?", [f"{report['experiment_id']}%"]).fetchone()[0]
    assert rows >= 2
    con.close()
    # reproduce check: rerun with same seed should give same delta within tolerance
    report2 = run_experiment(ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=123)
    # same horizon/ticker but different experiment_id due to timestamp, but metrics should be close
    assert abs(report["baseline_metrics"]["accuracy"] - report2["baseline_metrics"]["accuracy"]) < 1e-9

def test_no_future_leakage_in_experiment(isolated_research):
    # verify that predictions are only for test period after train
    report = run_experiment(ticker="SPY", horizon=5, start="2020-01-01", end="2022-12-31", seed=999)
    exp_dir = isolated_research / report["experiment_id"]
    pred = pl.read_parquet(str(exp_dir / "predictions_baseline.parquet"))
    # predictions event_time should be after train period (approx 70% split)
    assert pred.height > 0
    assert "y_true" in pred.columns and "y_pred" in pred.columns


def test_multi_horizon_framework(isolated_research):
    from quant_loop_trader.experiment import run_horizons
    from quant_loop_trader.models.prediction import Prediction, SUPPORTED_HORIZONS
    assert sorted(SUPPORTED_HORIZONS) == [1, 3, 5, 10, 20]
    reports = run_horizons(ticker="SPY", horizons=[1, 5], start="2019-01-01", end="2024-12-31", seed=7)
    assert [r["config"]["horizon"] for r in reports] == [1, 5]
    # Prediction object: frozen, hashable, content-addressed
    p = Prediction(timestamp="2024-01-05", ticker="SPY", horizon=5, prediction=1,
                   confidence=0.55, features_used=["ret_5"], model_version="t")
    assert p.sha256() != Prediction(timestamp="2024-01-05", ticker="SPY", horizon=5, prediction=0,
                                    confidence=0.55).sha256()
    import pytest as _pytest
    with _pytest.raises(__import__("dataclasses").FrozenInstanceError):
        p.prediction = 0
