"""Connector interfaces. Every connector returns (pl.DataFrame, source) where the
frame carries event_time + available_time Date columns — the PIT contract enforced
everywhere downstream by ReplayEngine/pit_filter."""
from __future__ import annotations

import polars as pl


def to_pit_frame(df: pl.DataFrame, event_col: str, available_col: str) -> pl.DataFrame:
    """Normalize a raw frame into the PIT contract: rename to event_time/available_time,
    cast to Date, sort, validate available >= event (a fact cannot be known before it happens)."""
    out = df.select(
        pl.col(event_col).cast(pl.Date).alias("event_time"),
        pl.col(available_col).cast(pl.Date).alias("available_time"),
        pl.exclude(event_col, available_col),
    ).sort("event_time")
    bad = out.filter(pl.col("available_time") < pl.col("event_time"))
    if bad.height:
        raise ValueError(f"leakage contract violated: {bad.height} rows with available_time < event_time")
    return out
