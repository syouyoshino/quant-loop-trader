"""SEC EDGAR fundamentals connector.

PIT semantics are exact here: every XBRL fact carries its period end (event_time)
and the filing date it became public (available_time). A Q2 report filed in August
is unavailable until August — enforced by data, not heuristics.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

from quant_loop_trader.connectors.common import to_pit_frame

load_dotenv()

logger = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# keep payloads small: core fundamental metrics only
TRACKED_TAGS = {
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NetIncomeLoss", "GrossProfit", "OperatingIncomeLoss",
    "Assets", "Liabilities", "StockholdersEquity",
    "EarningsPerShareDiluted", "CashAndCashEquivalentsAtCarryingValue",
}


def ticker_to_cik(ticker: str) -> int:
    r = requests.get(TICKERS_URL, headers={"User-Agent": _ua()}, timeout=30)
    r.raise_for_status()
    for _, row in r.json().items():
        if row["ticker"].upper() == ticker.upper():
            return int(row["cik_str"])
    raise ValueError(f"ticker {ticker} not found in SEC map")


def _ua() -> str:
    # SEC requires a declared identity; never embeds secrets
    return os.getenv("SEC_USER_AGENT", "quant-loop-trader research@example.com")


def fetch_company_facts(ticker: str, cache_dir: Path | None = None) -> tuple[pl.DataFrame, str]:
    """All tracked XBRL facts, long format. available_time = filing date (exact PIT).
    Raw JSON cached on disk (large, rate-limited API). Returns (df, 'sec_edgar')."""
    cik = ticker_to_cik(ticker)
    from quant_loop_trader.data import ROOT
    cache = (cache_dir or ROOT / "data" / "raw" / "sec")
    cache.mkdir(parents=True, exist_ok=True)
    raw_path = cache / f"{cik}.json"

    if raw_path.exists():
        facts_json = json.loads(raw_path.read_text())
        source = "sec_edgar_cache"
    else:
        r = requests.get(FACTS_URL.format(cik=cik), headers={"User-Agent": _ua()}, timeout=60)
        if r.status_code == 429:
            import time
            time.sleep(60)
            r = requests.get(FACTS_URL.format(cik=cik), headers={"User-Agent": _ua()}, timeout=60)
        r.raise_for_status()
        raw_path.write_text(r.text)
        facts_json = r.json()
        source = "sec_edgar"

    rows = []
    entity = facts_json.get("entityName", ticker)
    for taxonomy in ("us-gaap",):
        for tag, units in facts_json.get("facts", {}).get(taxonomy, {}).items():
            if tag not in TRACKED_TAGS:
                continue
            for unit_name, facts in units.get("units", {}).items():
                for f in facts:
                    rows.append({
                        "event_time": f["end"],                 # fiscal period end
                        "available_time": f["filed"],           # filing date — exact PIT
                        "ticker": ticker.upper(),
                        "metric": tag,
                        "value": float(f["val"]),
                        "form": f.get("form", ""),
                        "fiscal_year": f.get("fy"),
                        "fiscal_period": f.get("fp"),
                    })
    df = pl.DataFrame(rows) if rows else pl.DataFrame(schema={
        "event_time": pl.String, "available_time": pl.String})
    df = to_pit_frame(df, "event_time", "available_time")
    logger.info(f"sec {ticker}({entity}): {df.height} facts from {source}")
    return df, source
