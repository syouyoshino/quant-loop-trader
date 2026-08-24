import pytest

from quant_loop_trader.paper_trading import Order, PaperBroker, ExecutionSimulator, PortfolioTracker


def test_broker_disabled_by_default(monkeypatch):
    monkeypatch.delenv("QLT_PAPER_ENABLED", raising=False)
    with pytest.raises(RuntimeError, match="disabled"):
        PaperBroker(allow=True)  # even explicit allow needs env
    with pytest.raises(RuntimeError, match="disabled"):
        PaperBroker()  # and env alone is not enough (allow=False default)


def test_broker_enables_with_double_key(monkeypatch):
    monkeypatch.setenv("QLT_PAPER_ENABLED", "true")
    b = PaperBroker(allow=True)
    o = Order(timestamp="2024-01-05", ticker="SPY", side="buy", quantity=10)
    fill = b.submit(o, bar_close=470.0)
    assert fill["filled"] and fill["price"] > 470.0  # slippage applied on buys
    assert b.tracker.cash < 100_000
    assert b.tracker.positions["SPY"].quantity == 10


def test_order_validation_and_risk_guards():
    sim = ExecutionSimulator()
    bad = Order("2024-01-05", "SPY", "hold", -5)
    with pytest.raises(ValueError, match="invalid order"):
        sim.fill(bad, 100.0)
    t = PortfolioTracker(1_000)
    with pytest.raises(ValueError, match="insufficient cash"):
        t.execute(Order("2024-01-05", "SPY", "buy", 100), 50.0)
    with pytest.raises(ValueError, match="insufficient position"):
        t.execute(Order("2024-01-05", "SPY", "sell", 10), 50.0)


def test_round_trip_equity_accounts_for_costs():
    t = PortfolioTracker(10_000)
    t.execute(Order("d1", "SPY", "buy", 10), 100.0)
    eq_mid = t.equity({"SPY": 100.0})
    assert eq_mid == pytest.approx(10_000, abs=0.01)  # marked to market, no PnL yet
    t.execute(Order("d2", "SPY", "sell", 10), 110.0)
    assert t.equity({}) == pytest.approx(10_100)
