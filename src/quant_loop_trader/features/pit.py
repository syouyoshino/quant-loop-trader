"""PIT as-of joins: the core primitive for macro/fundamental features.

A value is usable at date t only if its available_time <= t. join_asof on
available_time gives exactly that, and truncation-invariance tests prove it.
"""
from __future__ import annotations

import polars as pl


def asof_values(obs: pl.DataFrame, dates: pl.DataFrame, value_col: str = "value") -> pl.DataFrame:
    """For each row of `dates` (column `date`), the latest observation whose
    available_time <= date. Null where nothing was available yet — never forward-fill
    across availability, only through time (which join_asof backward already encodes)."""
    obs = obs.select("available_time", value_col).sort("available_time")
    return (
        dates.select(pl.col("date").cast(pl.Date))
        .join_asof(obs.with_columns(pl.col("available_time").cast(pl.Date)),
                   left_on="date", right_on="available_time", strategy="backward")
        .rename({value_col: f"{value_col}_asof"})
    )


def yoy_growth(series: pl.Series) -> pl.Series:
    """Year-over-year growth of an annual series: v[t]/v[t-1] - 1."""
    vals = series.to_list()
    out = [None] * len(vals)
    for i in range(1, len(vals)):
        if vals[i] and vals[i - 1]:
            out[i] = vals[i] / vals[i - 1] - 1 if vals[i - 1] != 0 else None
    return pl.Series(out, dtype=pl.Float64)
