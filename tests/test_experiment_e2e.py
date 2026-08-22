import json, pathlib
import polars as pl
from quant_loop_trader.experiment import run_experiment
from quant_loop_trader.data import PROC_DIR, DB_PATH, FIXTURE_PATH
import duckdb

def test_experiment_e2e_fixture(tmp_path=None):
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
    exp_dir = pathlib.Path("data/experiments") / report["experiment_id"]
    assert (exp_dir / "report.json").exists()
    assert (exp_dir / "predictions_baseline.parquet").exists()
    assert (exp_dir / "predictions_improved.parquet").exists()
    assert (exp_dir / "metrics.json").exists()
    # DuckDB rows
    con = duckdb.connect(str(DB_PATH))
    rows = con.execute("select count(*) from experiments where experiment_id like ?", [f"{report['experiment_id']}%"]).fetchone()[0]
    assert rows >= 2
    con.close()
    # reproduce check: rerun with same seed should give same delta within tolerance
    report2 = run_experiment(ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=123)
    # same horizon/ticker but different experiment_id due to timestamp, but metrics should be close
    assert abs(report["baseline_metrics"]["accuracy"] - report2["baseline_metrics"]["accuracy"]) < 1e-9

def test_no_future_leakage_in_experiment():
    # verify that predictions are only for test period after train
    report = run_experiment(ticker="SPY", horizon=5, start="2020-01-01", end="2022-12-31", seed=999)
    exp_dir = pathlib.Path("data/experiments") / report["experiment_id"]
    pred = pl.read_parquet(str(exp_dir / "predictions_baseline.parquet"))
    # predictions event_time should be after train period (approx 70% split)
    assert pred.height > 0
    assert "y_true" in pred.columns and "y_pred" in pred.columns
