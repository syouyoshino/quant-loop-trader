"""Fundamental features from SEC XBRL facts.

PIT is exact: facts carry filing dates, and every computation below sees only
facts whose available_time <= date. Revenue/earnings growth use the two most
recent ANNUAL (10-K) figures known at that date.
"""
from __future__ import annotations

import polars as pl

FEATURE_NAMES = ["revenue_growth", "earnings_growth", "net_margin", "return_on_equity"]

ANNUAL_FORM = "10-K"


def _annual_asof(facts: pl.DataFrame, metric: str, date) -> float | None:
    """Latest 10-K value for metric filed on/before `date`."""
    f = (
        facts.filter((pl.col("metric") == metric) & (pl.col("form") == ANNUAL_FORM))
        .filter(pl.col("available_time") <= date)
        .sort("event_time")
    )
    return f["value"][-1] if f.height else None


def fundamental_features_at(facts: pl.DataFrame, ticker: str, date) -> dict:
    """Point-in-time fundamentals for one date. Missing data -> None (explicit)."""
    f = facts.filter(pl.col("ticker") == ticker.upper())
    rev = [_annual_asof(f, m, date) for m in ("Revenues",)]
    ni = [_annual_asof(f, m, date) for m in ("NetIncomeLoss",)]
    assets = _annual_asof(f, "Assets", date)
    equity = _annual_asof(f, "StockholdersEquity", date)

    def growth(vals: list[float | None]) -> float | None:
        if len(vals) >= 2 and all(v is not None for v in vals[:2]) and vals[1]:
            return vals[0] / vals[1] - 1
        return None

    # NOTE: revenue list holds only current annual; growth needs prior year — fetch both explicitly
    rev_prev = _prev_annual(f, "Revenues", date)
    ni_prev = _prev_annual(f, "NetIncomeLoss", date)
    rev_cur = rev[0]
    ni_cur = ni[0]

    return {
        "date": date,
        "ticker": ticker.upper(),
        "revenue_growth": (rev_cur / rev_prev - 1) if rev_cur and rev_prev else None,
        "earnings_growth": (ni_cur / ni_prev - 1) if ni_cur and ni_prev else None,
        "net_margin": (ni_cur / rev_cur) if ni_cur and rev_cur else None,
        "return_on_equity": (ni_cur / equity) if ni_cur and equity else None,
    }


def _prev_annual(facts: pl.DataFrame, metric: str, date) -> float | None:
    f = (
        facts.filter((pl.col("metric") == metric) & (pl.col("form") == ANNUAL_FORM))
        .filter(pl.col("available_time") <= date)
        # dedupe restatements: one row per fiscal period (latest filed wins)
        .unique(subset=["event_time"], keep="last")
        .sort("event_time")
    )
    return f["value"][-2] if f.height >= 2 else None


def build_fundamental_features(facts: pl.DataFrame, ticker: str, dates: pl.DataFrame) -> pl.DataFrame:
    facts = facts.with_columns(
        pl.col("event_time").cast(pl.Date), pl.col("available_time").cast(pl.Date),
    )
    rows = [fundamental_features_at(facts, ticker, d) for d in dates["date"].to_list()]
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).sort("date")
