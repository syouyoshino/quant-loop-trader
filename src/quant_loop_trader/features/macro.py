"""Macro features — interest rates, inflation, unemployment, regime indicators.

All values enter through as-of availability joins: a macro print published on the
15th is invisible to predictions dated the 14th, regardless of its period.
"""
from __future__ import annotations

import polars as pl

from quant_loop_trader.connectors import fred
from quant_loop_trader.features.pit import asof_values

SERIES = {
    "fed_funds": "DFF",       # interest rates (daily)
    "cpi": "CPIAUCSL",        # inflation level (monthly, SA)
    "unemployment": "UNRATE", # unemployment (monthly)
}

FEATURE_NAMES = ["fed_funds", "inflation_yoy", "unemployment", "high_rate_regime"]


def build_macro_features(dates: pl.DataFrame, obs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """dates: frame with `date` column. obs: keyed by feature name or FRED series id.
    Derived quantities (yoy, regimes) are computed on the FULL observation history
    first, then as-of joined — never on the sparse output rows."""
    out = dates.select(pl.col("date").cast(pl.Date))

    # interest rates + regime: computed on the daily history itself
    ff = _get(obs, "fed_funds", "DFF")
    if ff is not None:
        ff = ff.sort("available_time").with_columns(
            pl.col("value").rolling_median(window_size=505, min_periods=100).alias("_med"),
        ).with_columns(
            (pl.col("value") > pl.col("_med")).cast(pl.Int8).alias("high_rate_regime"),
        )
        out = out.with_columns(asof_values(ff.select(["available_time", "value"]), out, "value")["value_asof"].alias("fed_funds"))
        out = out.with_columns(asof_values(ff.select(["available_time", "high_rate_regime"]), out, "high_rate_regime")["high_rate_regime_asof"].fill_null(-1).alias("high_rate_regime"))

    # inflation: YoY computed on monthly level series (12-period shift), pub lag embedded
    cpi = _get(obs, "cpi", "CPIAUCSL")
    if cpi is not None:
        cpi = cpi.sort("available_time").with_columns(
            ((pl.col("value") / pl.col("value").shift(12) - 1) * 100).alias("inflation_yoy"),
        )
        out = out.with_columns(
            asof_values(cpi.select(["available_time", "inflation_yoy"]), out, "inflation_yoy")["inflation_yoy_asof"].alias("inflation_yoy"),
        )

    un = _get(obs, "unemployment", "UNRATE")
    if un is not None:
        out = out.with_columns(asof_values(un, out, "value")["value_asof"].alias("unemployment"))

    return out


def _get(obs: dict, name: str, sid: str) -> pl.DataFrame | None:
    return obs.get(name) if name in obs else obs.get(sid)


def fetch_observations(start: str, end: str) -> dict[str, pl.DataFrame]:
    return {name: fred.fetch_series(sid, start, end)[0] for name, sid in SERIES.items()}
