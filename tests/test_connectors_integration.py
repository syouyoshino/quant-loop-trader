"""Integration tests — hit live APIs, SKIP silently when credentials absent."""
import os

import pytest
from dotenv import load_dotenv

from quant_loop_trader.replay import pit_filter

load_dotenv()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("TIINGO_API_KEY", "").strip(), reason="no TIINGO key")
def test_tiingo_live_btcusd_daily_coverage(tmp_path, monkeypatch):
    """Prove the real Tiingo crypto contract matches the BTC research parser."""
    import quant_loop_trader.data as dm

    monkeypatch.setattr(dm, "PROC_DIR", tmp_path)
    df, source = dm.fetch_ohlcv(
        "BTCUSD", "2024-01-01", "2024-01-07", use_cache=False
    )
    assert source == "tiingo_crypto"
    assert df.height == 7
    assert str(df["event_time"].min()) == "2024-01-01"
    assert str(df["event_time"].max()) == "2024-01-07"
    assert df["event_time"].n_unique() == 7
    assert (df["available_time"] == df["event_time"]).all()
    dm.gap_check(df, ticker="BTCUSD")
    dm.coverage_check(df, "BTCUSD", "2024-01-01", "2024-01-07")


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ALPACA_API_KEY", "").strip(), reason="no ALPACA key")
def test_alpaca_live_bars():
    from quant_loop_trader.connectors import alpaca
    df, source = alpaca.fetch_bars("SPY", "2024-01-01", "2024-03-31")
    assert df.height > 30
    assert (df["available_time"] >= df["event_time"]).all()
    assert pit_filter(df, "2024-02-15")["event_time"].max() <= __import__("datetime").date(2024, 2, 15)


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("FRED_API_KEY", "").strip(), reason="no FRED key")
def test_fred_live_series():
    from quant_loop_trader.connectors import fred
    df, source = fred.fetch_series("DFF", "2023-01-01", "2023-12-31")
    assert df.height > 200  # daily federal funds rate
    assert (df["available_time"] > df["event_time"]).all()  # publication lag enforced


@pytest.mark.integration
def test_sec_live_company_facts():
    # SEC requires no key — only a User-Agent; run unless explicitly disabled
    if os.getenv("SKIP_SEC_INTEGRATION"):
        pytest.skip("SEC integration disabled")
    from quant_loop_trader.connectors import sec
    df, source = sec.fetch_company_facts("AAPL")
    assert df.height > 10
    # exact PIT: every fact available no earlier than its own filing date
    assert (df["available_time"] >= df["event_time"]).all()