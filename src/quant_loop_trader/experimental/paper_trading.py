"""Deferred offline paper-trading simulator.

This module is intentionally outside the active BTC research core until the
execution contract is finalized. It never touches a live broker endpoint.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Order:
    timestamp: str
    ticker: str
    side: str
    quantity: float
    order_type: str = "market"
    limit_price: float | None = None

    def validate(self) -> list[str]:
        errs = []
        if self.side not in ("buy", "sell"):
            errs.append("side must be buy|sell")
        if self.quantity <= 0:
            errs.append("quantity must be positive")
        if self.order_type == "limit" and self.limit_price is None:
            errs.append("limit order requires limit_price")
        return errs


@dataclass
class Position:
    ticker: str
    quantity: float = 0.0
    avg_price: float = 0.0

    def apply_fill(self, side: str, quantity: float, price: float) -> None:
        if side == "buy":
            total_cost = self.avg_price * self.quantity + price * quantity
            self.quantity += quantity
            self.avg_price = total_cost / self.quantity if self.quantity else 0.0
        elif side == "sell":
            self.quantity -= quantity
            if self.quantity <= 1e-9:
                self.quantity, self.avg_price = 0.0, 0.0


class ExecutionSimulator:
    """Fills against supplied historical closes. Offline by design."""

    def __init__(self, slippage_bps: float = 2.0):
        self.slippage_bps = slippage_bps

    def fill(self, order: Order, bar_close: float) -> dict:
        errs = order.validate()
        if errs:
            raise ValueError(f"invalid order: {errs}")
        if order.order_type not in ("market", "limit"):
            raise ValueError(f"unsupported order_type '{order.order_type}'")
        slip = bar_close * self.slippage_bps / 10_000
        if order.order_type == "limit":
            if order.side == "buy":
                mkt = bar_close + slip
                if mkt > order.limit_price:
                    return {"filled": False, "reason": "limit_below_market",
                            "limit": order.limit_price, "market": round(mkt, 4)}
                price = min(order.limit_price, mkt)
            else:
                mkt = bar_close - slip
                if mkt < order.limit_price:
                    return {"filled": False, "reason": "limit_above_market",
                            "limit": order.limit_price, "market": round(mkt, 4)}
                price = max(order.limit_price, mkt)
            return {"filled": True, "price": round(price, 4), "quantity": order.quantity,
                    "slippage_bps": self.slippage_bps}
        price = bar_close + slip if order.side == "buy" else bar_close - slip
        return {"filled": True, "price": round(price, 4), "quantity": order.quantity,
                "slippage_bps": self.slippage_bps}


class PortfolioTracker:
    def __init__(self, starting_cash: float = 100_000.0):
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.positions: dict[str, Position] = {}
        self.history: list[dict] = []

    def execute(self, order: Order, fill_price: float) -> None:
        cost = fill_price * order.quantity
        if order.side == "buy":
            if cost > self.cash:
                raise ValueError("insufficient cash")
            self.cash -= cost
            self.positions.setdefault(order.ticker, Position(order.ticker)).apply_fill(
                "buy", order.quantity, fill_price
            )
        else:
            pos = self.positions.setdefault(order.ticker, Position(order.ticker))
            if order.quantity > pos.quantity:
                raise ValueError("insufficient position")
            self.cash += cost
            pos.apply_fill("sell", order.quantity, fill_price)
        self.history.append({**asdict(order), "fill_price": fill_price,
                             "cash_after": round(self.cash, 2)})

    def equity(self, marks: dict[str, float]) -> float:
        pos_val = sum(
            p.quantity * marks.get(p.ticker, p.avg_price)
            for p in self.positions.values()
        )
        return round(self.cash + pos_val, 2)


class PaperBroker:
    enabled: bool = False

    def __init__(self, starting_cash: float = 100_000.0, allow: bool = False):
        env_ok = os.getenv("QLT_PAPER_ENABLED", "").lower() == "true"
        if not (allow and env_ok):
            raise RuntimeError(
                "PaperBroker disabled: requires allow=True AND QLT_PAPER_ENABLED=true"
            )
        self.simulator = ExecutionSimulator()
        self.tracker = PortfolioTracker(starting_cash)
        self.enabled = True

    def submit(self, order: Order, bar_close: float) -> dict:
        fill = self.simulator.fill(order, bar_close)
        if not fill.get("filled"):
            return {**fill, "cash_after": round(self.tracker.cash, 2)}
        self.tracker.execute(order, fill["price"])
        return {**fill, "cash_after": round(self.tracker.cash, 2)}
