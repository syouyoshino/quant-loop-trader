"""FRED / ALFRED macro connector.

PIT semantics: a macro observation for period P is NOT known on date P — it is
published later. We approximate available_time = period_end + publication lag
(monthly ~15d, quarterly ~45d, weekly ~7d, daily 1d).
ponytail: heuristic publication lags; upgrade path = ALFRED vintage_dates endpoint
which returns true real-time publication periods per revision.
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import polars as pl
import requests
from dotenv import load_dotenv

from quant_loop_trader.connectors.common import to_pit_frame

load_dotenv()

logger = logging.getLogger(__name__)

OBS_URL = "https://api.stlouisfed.org/fred/series/observations"
META_URL = "https://api.stlouisfed.org/fred/series"

# conservative publication lags by frequency (days after period end)
LAG_DAYS = {"Daily": 1, "Weekly": 7, "Monthly": 15, "Quarterly": 45, "Annual": 90}


def _frequency_lag(series_id: str, api_key: str) -> int:
    r = requests.get(META_URL, params={"series_id": series_id, "api_key": api_key, "file_type": "json"}, timeout=30)
    r.raise_for_status()
    freq = r.json()["seriess"][0].get("frequency", "Monthly")
    return LAG_DAYS.get(freq, 30)


def fetch_series(series_id: str, start: str, end: str) -> tuple[pl.DataFrame, str]:
    """Observations for a FRED series with PIT-safe availability dates.
    Returns (df, 'fred'). Columns: event_time (period), available_time (pub estimate),
    value, series_id."""
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("FRED_API_KEY not configured")

    lag = _frequency_lag(series_id, api_key)
    r = requests.get(OBS_URL, params={
        "series_id": series_id,
        "observation_start": start,
        "observation_end": end,
        "api_key": api_key,
        "file_type": "json",
    }, timeout=30)
    if r.status_code == 429:
        import time
        time.sleep(min(60, 60))
        r = requests.get(OBS_URL, params={
            "series_id": series_id, "observation_start": start, "observation_end": end,
            "api_key": api_key, "file_type": "json"}, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])

    rows = []
    for o in obs:
        if o["value"] == ".":  # missing placeholder
            continue
        event = date.fromisoformat(o["date"])
        rows.append({
            "event_time": event,
            "available_time": event + timedelta(days=lag),
            "value": float(o["value"]),
            "series_id": series_id,
        })
    df = pl.DataFrame(rows)
    if df.height == 0:
        return pl.DataFrame(schema={"event_time": pl.Date, "available_time": pl.Date, "value": pl.Float64, "series_id": pl.String}), "fred"
    logger.info(f"fred {series_id}: {df.height} obs, lag={lag}d")
    return to_pit_frame(df, "event_time", "available_time"), "fred"
