"""Phase 7 validation-expansion tests."""
import datetime

import numpy as np
import polars as pl

from quant_loop_trader.validation.holdout import apply_holdout, holdout_boundary


def _frame(n=200):
    return pl.DataFrame({
        "event_time": [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n)],
        "value": list(range(n)),
    })


def test_holdout_boundary_and_default_hiding():
    df = _frame()
    start, end = "2020-01-01", "2020-07-19"  # 200 days → boundary ~ day 170
    b = holdout_boundary(start, end)
    research = apply_holdout(df, start, end, use_holdout=False)
    assert research["event_time"].max() < datetime.date.fromisoformat(b)
    # hidden segment exists and is exactly the tail
    hold = apply_holdout(df, start, end, use_holdout=True)
    assert hold["event_time"].min() >= datetime.date.fromisoformat(b)
    assert research.height + hold.height == df.height


def test_ablation_identifies_signal_group():
    from quant_loop_trader.validation.ablation import run_ablation
    rng = np.random.default_rng(4)
    n = 600
    dates = pl.DataFrame({"event_time": [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n)]})
    signal = rng.normal(size=n)
    df = dates.with_columns([
        pl.Series("signal_feat", signal),
        pl.Series("noise_a", rng.normal(size=n)),
        pl.Series("noise_b", rng.normal(size=n)),
        pl.Series("label", (signal > 0).astype(int)),
        pl.Series("available_time", pl.Series(dates["event_time"])),
        pl.Series("close", np.full(n, 100.0)),
    ])
    import quant_loop_trader.experiment as exp_mod

    orig = exp_mod.build_train_test
    def fake_build(ticker, start, end, horizon, feature_fn, feat_cols):
        clean = exp_mod.time_split(df, 0.7, purge=horizon)
        return clean
    exp_mod.build_train_test = fake_build
    try:
        groups = {"signal": ["signal_feat"], "noiseA": ["noise_a"], "noiseB": ["noise_b"]}
        out = run_ablation("SPY", "2020-01-01", "2021-08-01", 5, 42, groups,
                           model_builder=lambda: __import__(
                               "quant_loop_trader.models.registry", fromlist=["LogisticModel"]
                           ).LogisticModel(seed=42),
                           feature_fn=lambda df: df,
                           feat_cols=["signal_feat", "noise_a", "noise_b"])
        sig_delta = out.filter(pl.col("removed") == "signal")["delta_vs_full"][0]
        noise_delta = out.filter(pl.col("removed") == "noiseA")["delta_vs_full"][0]
        # removing the signal group must hurt; removing noise must not help/hurt much
        assert sig_delta < -0.2
        assert abs(noise_delta) < sig_delta * -1 or noise_delta >= sig_delta
    finally:
        exp_mod.build_train_test = orig


def test_bias_dashboard_aggregates(isolated_research):
    from quant_loop_trader.experiment import run_experiment
    from quant_loop_trader.validation.dashboard import build_dashboard
    run_experiment(ticker="SPY", horizon=5, start="2019-01-01", end="2024-12-31", seed=21)
    d = build_dashboard(isolated_research)
    for key in ["experiments_total", "decisions", "keep_ratio", "hypotheses_tested",
                "gate_rejection_reasons", "degenerate_flags", "verdict"]:
        assert key in d
    assert d["experiments_total"] >= 1
    assert d["verdict"] in ("HEALTHY_SKEPTICAL", "REVIEW", "SUSPICIOUS_KEEP_RATE",
                            "COMPROMISED", "NO_RESEARCH_YET")
