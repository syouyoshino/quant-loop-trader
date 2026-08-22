import polars as pl
from quant_loop_trader.features import add_features, feature_columns
from quant_loop_trader.data import PROC_DIR

def test_features_lagged_no_leakage():
    df = pl.read_parquet(str(PROC_DIR / "SPY.parquet"))
    df_feat = add_features(df)
    # features at t should not use close[t]; check that ret_1 at row 10 equals close[9]/close[8]-1 not close[10]/close[9]
    # shifting ensures first valid feature appears after window
    assert df_feat["ret_1"].null_count() > 0
    # ensure no label leakage column exists
    assert "fwd_ret" not in df_feat.columns
    # all feature columns exist
    for c in feature_columns():
        assert c in df_feat.columns

def test_features_shifted_one_bar():
    # construct tiny frame to verify shift
    df = pl.DataFrame({
        "event_time": pl.date_range(pl.datetime(2020,1,1), pl.datetime(2020,1,10), "1d", eager=True).alias("d"),
        "available_time": pl.date_range(pl.datetime(2020,1,1), pl.datetime(2020,1,10), "1d", eager=True).alias("d2"),
        "open": [1.0]*10, "high":[1.0]*10, "low":[1.0]*10, "close": [100,101,102,103,104,105,106,107,108,109], "volume":[1]*10
    })
    # need to cast date
    df = df.with_columns(pl.col("event_time").cast(pl.Date), pl.col("available_time").cast(pl.Date))
    feat = add_features(df)
    # ret_1 at index 2 should be close[1]/close[0]-1 = 0.01, but shifted by 1 so at index2 ret_1 = close[1]/close[0]-1
    # ret_1 at index 0,1 should be null due to shift
    vals = feat["ret_1"].to_list()
    assert vals[0] is None and vals[1] is None
    assert abs(vals[2] - 0.01) < 1e-6


def test_features_truncation_invariant():
    """No feature at time t may read rows > t: recomputing on any truncated
    history must reproduce the same final row, or lookahead exists."""
    from quant_loop_trader.features import add_improved_features
    df = pl.read_parquet(str(PROC_DIR / "SPY.parquet")).sort("event_time")
    full = add_improved_features(df)
    for t in [20, 100, len(df) // 2, len(df) - 1]:
        partial = add_improved_features(df.slice(0, t))
        assert partial.row(-1) == full.row(t - 1), f"lookahead detected at index {t}"
