
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


def test_deflated_sharpe_penalizes_many_trials():
    lucky = deflated_sharpe_ratio(sharpe=2.1, n_obs=500, n_trials=1000)
    lone = deflated_sharpe_ratio(sharpe=2.1, n_obs=500, n_trials=1)
    assert lucky["dsr"] < lone["dsr"]                       # same Sharpe, less believable after mining
    assert lucky["verdict"] != "GENUINE" or lone["dsr"] > lucky["dsr"]
    assert deflated_sharpe_ratio(-1.0, 500, 5)["verdict"] == "PROBABLY_LUCK"
    assert family_wise_stats({"total": 100, "successes": 3, "failures": 97})["failed_candidates"] == 97
