"""ReplayEngine — PIT enforcement. Only evaluation may see future."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def _to_date(ts) -> date:
    if isinstance(ts, date) and not isinstance(ts, datetime):
        return ts
    if isinstance(ts, datetime):
        return ts.date()
    if isinstance(ts, str):
        return date.fromisoformat(str(ts)[:10])
    raise ValueError(f"unsupported timestamp {ts!r}")


def pit_filter(df: pl.DataFrame, timestamp) -> pl.DataFrame:
    """PIT filter for ANY connector frame carrying available_time — macro, fundamentals,
    news. Same contract as ReplayEngine.get_snapshot for arbitrary frames."""
    ts = _to_date(timestamp)
    return df.filter(pl.col("available_time") <= ts).sort("event_time")


class ReplayEngine:
    """Reconstruct information state at prediction time.

    ``ticker`` is explicit when the parquet filename is content-addressed rather
    than ticker-named. This lets immutable dataset snapshots remain the canonical
    post-acquisition input without weakening ticker validation.
    """

    def __init__(self, parquet_path: str | Path, ticker: str | None = None):
        p = Path(parquet_path)
        if not p.exists():
            raise FileNotFoundError(f"parquet not found: {p}")
        df = pl.read_parquet(str(p))
        needed = {"event_time", "available_time", "close"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"missing columns {missing} in {p}")
        for c in ("event_time", "available_time"):
            if df[c].dtype != pl.Date:
                df = df.with_columns(pl.col(c).cast(pl.Date))
        self.df = df.sort("event_time")
        self.ticker = str(ticker or p.stem)
        logger.info(json.dumps({"event": "replay_loaded", "rows": self.df.height,
                                "path": str(p), "ticker": self.ticker}))

    def get_snapshot(self, ticker: str, timestamp) -> pl.DataFrame:
        """Return rows where available_time <= timestamp, with ticker validation."""
        if ticker != self.ticker:
            raise ValueError(f"engine holds {self.ticker}, requested {ticker}")
        ts = _to_date(timestamp)
        snap = self.df.filter(pl.col("available_time") <= ts)
        if snap.height > 0:
            max_av = snap["available_time"].max()
            assert max_av <= ts, f"leakage: max available {max_av} > {ts}"
        return snap

    def full_history(self) -> pl.DataFrame:
        return self.df

    def evaluate_future(self, ticker: str, timestamp) -> pl.DataFrame:
        """Outcomes strictly AFTER timestamp. For EVALUATION systems only."""
        if ticker != self.ticker:
            raise ValueError(f"engine holds {self.ticker}, requested {ticker}")
        ts = _to_date(timestamp)
        return self.df.filter(pl.col("event_time") > ts).sort("event_time")
