"""Paper trading PREPARATION (Phase 11). Interfaces + offline simulator only.

SAFETY BOUNDARY (charter):
- No real orders. No live endpoints. The broker below is an OFFLINE simulator
  filling against historical bars.
- PaperBroker refuses to construct unless explicitly allowed: allow=True AND
  env QLT_PAPER_ENABLED=true. Default state is disabled everywhere.
- Even when enabled, this simulates fills for research feedback; it never
  touches Alpaca's order API. That integration point is deliberately absent.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Order:
    """Order abstraction — strategy-neutral."""
    timestamp: str
    ticker: str
    side: str                 # buy | sell
    quantity: float
    order_type: str = "market"  # market | limit
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
    """Fills market orders at the given bar close + slippage bps. Offline by design."""

    def __init__(self, slippage_bps: float = 2.0):
        self.slippage_bps = slippage_bps

    def fill(self, order: Order, bar_close: float) -> dict:
        errs = order.validate()
        if errs:
            raise ValueError(f"invalid order: {errs}")
        slip = bar_close * self.slippage_bps / 10_000
        price = bar_close + slip if order.side == "buy" else bar_close - slip
        return {"filled": True, "price": round(price, 4), "quantity": order.quantity,
                "slippage_bps": self.slippage_bps}


class PortfolioTracker:
    """Positions + cash ledger for the simulation."""

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
            self.positions.setdefault(order.ticker, Position(order.ticker)).apply_fill("buy", order.quantity, fill_price)
        else:
            pos = self.positions.setdefault(order.ticker, Position(order.ticker))
            if order.quantity > pos.quantity:
                raise ValueError("insufficient position")
            self.cash += cost
            pos.apply_fill("sell", order.quantity, fill_price)
        self.history.append({**asdict(order), "fill_price": fill_price,
                             "cash_after": round(self.cash, 2)})

    def equity(self, marks: dict[str, float]) -> float:
        pos_val = sum(p.quantity * marks.get(p.ticker, p.avg_price) for p in self.positions.values())
        return round(self.cash + pos_val, 2)


class PaperBroker:
    """Facade wiring simulator + tracker. DISABLED BY DEFAULT (see module docstring)."""

    enabled: bool = False

    def __init__(self, starting_cash: float = 100_000.0, allow: bool = False):
        env_ok = os.getenv("QLT_PAPER_ENABLED", "").lower() == "true"
        if not (allow and env_ok):
            raise RuntimeError(
                "PaperBroker disabled: requires allow=True AND QLT_PAPER_ENABLED=true "
                "(safety boundary — paper trading stays off until explicitly activated)"
            )
        self.simulator = ExecutionSimulator()
        self.tracker = PortfolioTracker(starting_cash)

    def submit(self, order: Order, bar_close: float) -> dict:
        fill = self.simulator.fill(order, bar_close)
        self.tracker.execute(order, fill["price"])
        return fill
