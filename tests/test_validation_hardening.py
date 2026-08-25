
import numpy as np

from quant_loop_trader.validation.walkforward import WalkForwardValidator, make_folds
from quant_loop_trader.validation.multiple_testing import (
    benjamini_hochberg, deflated_sharpe_ratio, family_wise_stats,
)


def _wf_frame(n=800, seed=5):
    rng = np.random.default_rng(seed)
    dates = [datetime.date(2019, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    sig = rng.normal(size=n)
    return pl.DataFrame({
        "event_time": dates,
        "available_time": dates,
        "signal_feat": sig,
        "noise_feat": rng.normal(size=n),
        "close": list(100 * np.cumprod(1 + rng.normal(0.0002, 0.01, n))),
        "label": (sig > 0).astype(int),
    })


import datetime
import polars as pl


def test_make_folds_expanding_and_contiguous():
    folds = make_folds(list(range(400)), n_folds=4, min_train_frac=0.5)
    assert len(folds) == 4
    prev_val_end = None
    for f in folds:
        a, b = f["train_idx"]
        c, d = f["validation_idx"]
        assert a == 0 and b == c and c < d          # expanding train, adjacent validation
        if prev_val_end is not None:
            assert c == prev_val_end                 # no gaps, no overlaps
        prev_val_end = d


def test_walkforward_persists_across_folds():
    from quant_loop_trader.models.registry import LogisticModel
    df = _wf_frame()
    v = WalkForwardValidator(lambda: LogisticModel(seed=1), n_folds=3)
    out = v.run(df, ["signal_feat", "noise_feat"])
    assert len(out["folds"]) == 3
    for fold in out["folds"]:
        # PIT: training period strictly before validation period
        assert fold["training_period"][1] <= fold["validation_period"][0]
        assert {"window_id", "performance", "risk_metrics", "prediction_accuracy"} <= set(fold)
    assert out["mean_accuracy"] > 0.6      # signal is learnable in every fold
    assert out["accuracy_dispersion"] < 0.15
    assert out["stable_across_time"] is True


def test_bh_fdr_controls_familywise_error():
    pvals = [0.001, 0.008, 0.039, 0.041, 0.2, 0.45, 0.9]
    rejects = benjamini_hochberg(pvals, fdr=0.10)
    assert rejects[:2] == [True, True]     # strong findings survive
    assert not rejects[-1]                 # weak finding rejected


def test_deflated_sharpe_penalizes_dispersion():
    # many dispersed trials → higher E[max] under H0 → lower DSR for same Sharpe
    few = deflated_sharpe_ratio(1.2, n_obs=250, n_trials=[0.8, 1.2])
    many = deflated_sharpe_ratio(1.2, n_obs=250,
                                 n_trials=[-1.0, 0.1, 0.5, 0.9, 1.2, 1.6, 2.2])
    assert many["dsr"] <= few["dsr"]
    assert deflated_sharpe_ratio(-1.0, 500, [0.5, -1.0])["verdict"] == "PROBABLY_LUCK"
    assert family_wise_stats({"total": 100, "successes": 3, "failures": 97})["failed_candidates"] == 97


def test_dsr_uses_empirical_dispersion_not_count():
    """Audit round-2: same count with different dispersions must give different DSR;
    n_trials=1 must NOT fabricate a negative deflation threshold."""
    from quant_loop_trader.validation.multiple_testing import deflated_sharpe_ratio
    tight = deflated_sharpe_ratio(1.5, n_obs=500, n_trials=[1.4, 1.5, 1.6])
    wide = deflated_sharpe_ratio(1.5, n_obs=500, n_trials=[0.0, 1.5, 3.0])
    assert wide["dsr"] < tight["dsr"]  # more dispersion → less believable
    single = deflated_sharpe_ratio(2.1, n_obs=500, n_trials=1)
    assert single["expected_max_sharpe_h0"] == 0.0      # no fabricated negative threshold
    assert single["verdict"] == "GENUINE"               # plain PSR at high Sharpe
    count_only = deflated_sharpe_ratio(2.1, n_obs=500, n_trials=50)
    assert count_only["dispersion_note"] == "count_only_no_dispersion"
    assert count_only["expected_max_sharpe_h0"] == 0.0


def test_dsr_luck_still_flagged_with_empirical_input():
    from quant_loop_trader.validation.multiple_testing import deflated_sharpe_ratio
    # short observation window + dispersed trials: Sharpe 1.0 cannot clear the bar
    short_lived = deflated_sharpe_ratio(1.0, n_obs=60,
                                        n_trials=[-1.0, 0.2, 0.4, 1.0, 0.9, 0.1, -0.3, 0.6])
    # dispersed trials raise E[max] to ~0.96 → DSR collapses from certain to coin-flip
    assert short_lived["verdict"] == "LOW_CONFIDENCE" and short_lived["dsr"] < 0.7
    # a lone identical run (no deflation) stays confident — proving dispersion bites
    lone = deflated_sharpe_ratio(1.0, n_obs=60, n_trials=[1.0])
    assert lone["dsr"] > short_lived["dsr"] and lone["expected_max_sharpe_h0"] == 0.0
