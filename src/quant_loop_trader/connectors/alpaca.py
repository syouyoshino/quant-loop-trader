"""Alpaca Market Data connector — historical daily bars only. No trading, no orders."""
from __future__ import annotations

import logging
import os

import polars as pl
import requests
from dotenv import load_dotenv

from quant_loop_trader.connectors.common import to_pit_frame

load_dotenv()

logger = logging.getLogger(__name__)

BARS_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"


def fetch_bars(symbol: str, start: str, end: str, feed: str = "iex", limit: int = 10000) -> tuple[pl.DataFrame, str]:
    """Daily OHLCV bars. event_time/available_time = bar date (daily bar is final at close).
    Uses IEX feed by default (free tier). Returns (df, 'alpaca')."""
    key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not configured")

    rows: list[dict] = []
    page_token = None
    while True:
        params = {
            "timeframe": "1Day",
            "start": start,
            "end": end,
            "feed": feed,
            "limit": min(limit, 10000),
            "adjustment": "split",
        }
        if page_token:
            params["page_token"] = page_token
        r = requests.get(
            BARS_URL.format(symbol=symbol),
            params=params,
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=30,
        )
        if r.status_code == 429:
            import time
            time.sleep(min(60, 2 ** 2))
            continue
        r.raise_for_status()
        payload = r.json()
        rows.extend(payload.get("bars", []))
        page_token = payload.get("next_page_token")
        if not page_token:
            break

    if not rows:
        return pl.DataFrame(schema={"event_time": pl.Date, "available_time": pl.Date}), "alpaca"

    df = pl.DataFrame(rows)
    # alpaca bar: {"t": "...T04:00:00Z", "o","h","l","c","v", ...} t = bar open timestamp (UTC)
    df = df.select(
        pl.col("t").str.slice(0, 10).alias("event_time"),
        pl.col("o").alias("open"), pl.col("h").alias("high"), pl.col("l").alias("low"),
        pl.col("c").alias("close"), pl.col("v").cast(pl.Int64).alias("volume"),
    )
    df = df.with_columns(pl.col("available_time").alias("available_time")) if "available_time" in df.columns else df.with_columns(pl.col("event_time").alias("available_time"))
    return to_pit_frame(df, "event_time", "available_time"), "alpaca"
