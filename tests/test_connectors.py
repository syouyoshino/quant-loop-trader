"""Unit tests for connectors — all HTTP mocked, zero network."""
import json
from pathlib import Path

import polars as pl
import pytest

from quant_loop_trader.connectors import alpaca, fred, sec
from quant_loop_trader.connectors.common import to_pit_frame
from quant_loop_trader.replay import pit_filter

FIX = Path(__file__).parent / "fixtures" / "connectors"


def _mock(monkeypatch, url_substring: str, payload_path: str):
    payload = (FIX / payload_path).read_text()

    class R:
        status_code = 200
        text = payload
        def raise_for_status(self): ...
        def json(self): return json.loads(payload)

    calls = []
    def fake_get(url, **kwargs):
        calls.append(url)
        return R()
    monkeypatch.setattr("requests.get", fake_get)
    return calls


# --- PIT contract -----------------------------------------------------------

def test_to_pit_frame_normalizes_and_validates():
    df = pl.DataFrame({"event_time": ["2024-01-02"], "available_time": ["2024-01-05"], "value": [1.0]})
    out = to_pit_frame(df, "event_time", "available_time")
    assert out["event_time"].dtype == pl.Date and out["available_time"].dtype == pl.Date

def test_to_pit_frame_rejects_knowledge_before_event():
    df = pl.DataFrame({"event_time": ["2024-01-05"], "available_time": ["2024-01-01"], "value": [1.0]})
    with pytest.raises(ValueError, match="leakage contract"):
        to_pit_frame(df, "event_time", "available_time")


# --- Alpaca -----------------------------------------------------------------

def test_alpaca_bars_parse_and_pit(monkeypatch):
    _mock(monkeypatch, "alpaca.markets", "alpaca_bars.json")
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    df, source = alpaca.fetch_bars("SPY", "2024-01-01", "2024-01-31")
    assert source == "alpaca"
    assert df.height == 2
    assert {"event_time", "available_time", "close", "volume"} <= set(df.columns)
    # daily bars: available same day as event
    assert (df["available_time"] == df["event_time"]).all()


def test_alpaca_requires_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        alpaca.fetch_bars("SPY", "2024-01-01", "2024-01-31")


# --- FRED -------------------------------------------------------------------

def test_fred_observations_publication_lag(monkeypatch):
    _mock(monkeypatch, "stlouisfed.org/fred/series", "fred_obs.json")
    _mock(monkeypatch, "stlouisfed.org/alfred", "fred_meta.json")  # not used but harmless
    seen = []
    orig = __import__("requests").get
    def routed(url, **kw):
        seen.append(url)
        if "observations" not in url:
            payload = json.loads((FIX / "fred_meta.json").read_text())
        else:
            payload = json.loads((FIX / "fred_obs.json").read_text())
        class R:
            status_code = 200
            def raise_for_status(self): ...
            def json(self): return payload
        return R()
    monkeypatch.setattr("requests.get", routed)
    monkeypatch.setenv("FRED_API_KEY", "k")
    df, source = fred.fetch_series("DFF", "2024-01-01", "2024-03-31")
    assert source == "fred"
    assert df.height == 2  # "." placeholder dropped
    row = df.filter(pl.col("event_time") == pl.date(2024, 1, 1)).row(0)
    # monthly series published ~15d after period end — never on the period date itself
    assert row[1] > row[0]
    # PIT: nothing available before its own publication
    assert pit_filter(df, "2024-01-10").height == 0
    assert pit_filter(df, "2024-01-20").height == 1


def test_fred_requires_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        fred.fetch_series("DFF", "2024-01-01", "2024-03-31")


# --- SEC EDGAR --------------------------------------------------------------

def test_sec_facts_exact_pit_from_filing_dates(monkeypatch, tmp_path):
    _mock(monkeypatch, "company_tickers.json", "sec_tickers.json")

    calls = []
    orig = __import__("requests").get
    def routed(url, **kw):
        calls.append(url)
        payload = (FIX / ("sec_tickers.json" if "tickers" in url else "sec_facts.json")).read_text()
        class R:
            status_code = 200
            text = payload
            def raise_for_status(self): ...
            def json(self): return json.loads(payload)
        return R()
    monkeypatch.setattr("requests.get", routed)

    df, source = sec.fetch_company_facts("AAPL", cache_dir=tmp_path / "cache")
    assert source == "sec_edgar"
    assert df.height == 3
    assert {"metric", "value", "form"} <= set(df.columns)
    # THE core PIT property: Q2 facts (period end 2024-06-30) only available at filing (2024-08-02)
    june = df.filter(pl.col("event_time") == pl.date(2024, 6, 30))
    import datetime
    assert (june["available_time"] >= datetime.date(2024, 8, 2)).all()
    # replay compatibility: snapshot before filing date sees NOTHING from that quarter
    import datetime
    assert pit_filter(june, "2024-08-01").height == 0
    assert pit_filter(june, "2024-08-03").height == 2


def test_sec_caches_raw_payload(monkeypatch, tmp_path):
    _mock(monkeypatch, "company_tickers.json", "sec_tickers.json")
    n = [0]
    orig = __import__("requests").get
    def routed(url, **kw):
        n[0] += 1
        payload = (FIX / ("sec_tickers.json" if "tickers" in url else "sec_facts.json")).read_text()
        class R:
            status_code = 200
            text = payload
            def raise_for_status(self): ...
            def json(self): return json.loads(payload)
        return R()
    monkeypatch.setattr("requests.get", routed)
    cache = tmp_path / "c"
    sec.fetch_company_facts("AAPL", cache_dir=cache)
    raw = list(cache.glob("*.json"))
    assert len(raw) == 1
    # second call serves from cache: no facts request fired
    sec.fetch_company_facts("AAPL", cache_dir=cache)
    assert n[0] == 3  # tickers map fetched per call (uncached); facts payload served from disk cache
