"""Out-of-time holdout (Phase 7): the final segment of every window is PERMANENTLY
hidden from training/tuning/research splits. Only an explicit holdout evaluation
may touch it."""
from __future__ import annotations

from datetime import datetime

HOLDOUT_FRACTION = 0.15


def holdout_boundary(start: str, end: str) -> str:
    """ISO date where the hidden region begins for a [start, end] research window."""
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    days = int((e - s).days * HOLDOUT_FRACTION)
    return (e - __import__("datetime").timedelta(days=days)).date().isoformat()


def apply_holdout(df, start: str, end: str, use_holdout: bool):
    """use_holdout=False (research default): drop rows at/after the boundary so no
    model, feature selection, or tuning ever sees them.
    use_holdout=True (final evaluation ONLY): return just the hidden segment as test."""
    import polars as pl
    b = pl.lit(holdout_boundary(start, end)).str.strptime(pl.Date, "%Y-%m-%d")
    if use_holdout:
        return df.filter(pl.col("event_time") >= b)
    return df.filter(pl.col("event_time") < b)
