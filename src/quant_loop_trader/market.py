"""Canonical market semantics shared by research, validation, and dashboard."""
from __future__ import annotations

import os
from datetime import date

CRYPTO_SYMBOLS = {
    "BTC", "ETH", "BTCUSD", "ETHUSD", "BTC-USD", "ETH-USD", "BTCUSDT",
}
TRADING_DAYS = 252
CRYPTO_DAYS = 365
DEFAULT_CRYPTO_HOLDOUT_START = "2024-01-01"
DEFAULT_CRYPTO_CAMPAIGN_ID = "btc_pre2024_v1"


def normalize_ticker(ticker: str | None) -> str:
    return (ticker or "").upper().replace("/", "").strip()


def is_crypto(ticker: str | None) -> bool:
    return normalize_ticker(ticker) in CRYPTO_SYMBOLS


def calendar_days(ticker: str | None) -> int:
    return CRYPTO_DAYS if is_crypto(ticker) else TRADING_DAYS


def periods_per_year(ticker: str | None, horizon: int) -> float:
    return calendar_days(ticker) / max(1, int(horizon))


def campaign_holdout_start(ticker: str | None) -> str | None:
    """Return a fixed, campaign-level holdout boundary when one is configured.

    Crypto research defaults to a permanent 2024-01-01 boundary so changing an
    experiment's end date cannot recycle previously hidden observations back into
    research. Set QLT_CRYPTO_HOLDOUT_START to override it for a deliberately new
    campaign. Equities retain the legacy per-experiment fractional holdout unless
    QLT_HOLDOUT_START is set.
    """
    if is_crypto(ticker):
        raw = os.getenv("QLT_CRYPTO_HOLDOUT_START", DEFAULT_CRYPTO_HOLDOUT_START)
    else:
        raw = os.getenv("QLT_HOLDOUT_START", "")
    raw = raw.strip()
    if not raw:
        return None
    date.fromisoformat(raw)  # fail fast on invalid configuration
    return raw


def campaign_id(ticker: str | None) -> str:
    """Stable research-campaign identity bound to experiment fingerprints.

    A crypto holdout boundary is part of the scientific design. Moving it changes
    which observations are available to research, so the move must be explicit and
    must use a new campaign id. This prevents a 2026 campaign from sharing an
    experiment identity with the original pre-2024 campaign.
    """
    if not is_crypto(ticker):
        return (os.getenv("QLT_CAMPAIGN_ID", "default").strip() or "default")

    cid = os.getenv("QLT_CRYPTO_CAMPAIGN_ID", DEFAULT_CRYPTO_CAMPAIGN_ID).strip()
    if not cid:
        raise ValueError("crypto_campaign_id_empty")
    boundary = campaign_holdout_start(ticker)
    if (
        boundary != DEFAULT_CRYPTO_HOLDOUT_START
        and cid == DEFAULT_CRYPTO_CAMPAIGN_ID
    ):
        raise ValueError(
            "custom_crypto_holdout_requires_new_campaign_id:"
            "set QLT_CRYPTO_CAMPAIGN_ID"
        )
    return cid