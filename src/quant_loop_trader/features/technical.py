"""Feature engineering — all lagged, no future lookahead."""
from __future__ import annotations

import polars as pl


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add baseline features. All computed from past bars only (shifted)."""
    if df.height == 0:
        return df
    df = df.sort("event_time")
    # ensure close exists
    # returns
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret_1_raw"),
        (pl.col("close") / pl.col("close").shift(5) - 1).alias("ret_5_raw"),
    )
    # SMA10 gap
    df = df.with_columns(
        pl.col("close").rolling_mean(window_size=10).alias("sma10"),
    )
    df = df.with_columns(
        ((pl.col("close") - pl.col("sma10")) / pl.col("close")).alias("ma10_gap_raw"),
        pl.col("ret_1_raw").rolling_std(window_size=10).alias("vol10_raw"),
    )
    # RSI14 Wilder
    # gains/losses from ret? Use price diff
    df = df.with_columns(
        (pl.col("close") - pl.col("close").shift(1)).alias("delta"),
    )
    df = df.with_columns(
        pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0).alias("gain"),
        pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0).alias("loss"),
    )
    df = df.with_columns(
        pl.col("gain").rolling_mean(window_size=14).alias("avg_gain"),  # simple mean; NOT Wilder — registry documents this
        pl.col("loss").rolling_mean(window_size=14).alias("avg_loss"),
    )
    df = df.with_columns(
        (100 - (100 / (1 + pl.col("avg_gain") / (pl.col("avg_loss") + 1e-12)))).alias("rsi14_raw"),
    )
    # shift all raw features by 1 to ensure PIT: feature at t uses info up to t-1 only
    # available_time == event_time for L1, so shifting prevents using close[t] to predict t
    for c in ["ret_1_raw", "ret_5_raw", "ma10_gap_raw", "vol10_raw", "rsi14_raw"]:
        df = df.with_columns(pl.col(c).shift(1).alias(c.replace("_raw", "")))

    # drop intermediate
    df = df.drop(["ret_1_raw", "ret_5_raw", "ma10_gap_raw", "vol10_raw", "rsi14_raw", "sma10", "delta", "gain", "loss", "avg_gain", "avg_loss"])
    # feature list for model
    return df


def feature_columns() -> list[str]:
    return ["ret_1", "ret_5", "ma10_gap", "vol10", "rsi14"]


def add_improved_features(df: pl.DataFrame) -> pl.DataFrame:
    """Improved hypothesis: vol regime interaction."""
    df = add_features(df)
    # vol regime: interaction term ret_5 * vol10 (trend persistence varies by vol)
    df = df.with_columns(
        (pl.col("ret_5") * pl.col("vol10")).alias("ret5_x_vol10"),
        (pl.col("ret_5") / (pl.col("vol10") + 1e-9)).alias("ret5_div_vol10"),
    )
    return df


def improved_feature_columns() -> list[str]:
    return feature_columns() + ["ret5_x_vol10", "ret5_div_vol10"]
