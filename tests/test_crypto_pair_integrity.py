import pytest

import quant_loop_trader.data as dm


def _bar(close: float) -> dict:
    return {
        "date": "2024-01-01T00:00:00.000Z",
        "open": close - 1,
        "high": close + 1,
        "low": close - 2,
        "close": close,
        "volume": 10.0,
    }


def test_crypto_parser_selects_requested_pair_not_first_response_row():
    rows = [
        {"ticker": "ethusd", "priceData": [_bar(2_500.0)]},
        {"ticker": "btcusd", "priceData": [_bar(42_500.0)]},
    ]

    df = dm._parse_tiingo_crypto(rows, "BTCUSD")

    assert df.height == 1
    assert df["close"][0] == 42_500.0


def test_crypto_parser_rejects_response_without_requested_pair():
    rows = [{"ticker": "ethusd", "priceData": [_bar(2_500.0)]}]

    with pytest.raises(ValueError, match="requested_crypto_pair_missing:BTCUSD"):
        dm._parse_tiingo_crypto(rows, "BTCUSD")


def test_crypto_parser_rejects_ambiguous_duplicate_requested_pair():
    rows = [
        {"ticker": "btcusd", "priceData": [_bar(42_500.0)]},
        {"ticker": "BTC-USD", "priceData": [_bar(42_600.0)]},
    ]

    with pytest.raises(ValueError, match="requested_crypto_pair_ambiguous:BTCUSD:matches=2"):
        dm._parse_tiingo_crypto(rows, "BTCUSD")


def test_fetch_crypto_does_not_cache_wrong_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "PROC_DIR", tmp_path)
    monkeypatch.setenv("TIINGO_API_KEY", "secret")
    monkeypatch.setattr(
        dm,
        "_tiingo_fetch",
        lambda *_args, **_kwargs: [{"ticker": "ethusd", "priceData": [_bar(2_500.0)]}],
    )

    with pytest.raises(RuntimeError, match="Tiingo failed"):
        dm.fetch_ohlcv("BTCUSD", "2024-01-01", "2024-01-01", use_cache=False)

    assert not (tmp_path / "BTCUSD.parquet").exists()
