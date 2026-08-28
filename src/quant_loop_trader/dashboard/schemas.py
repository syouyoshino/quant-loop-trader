"""Shapes and vocabulary shared by the dashboard service and API.

No metric is invented here. ``NA`` is the single sentinel for "Quant Loop has
not produced this value"; the frontend renders it as N/A.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from quant_loop_trader.market import (
    CRYPTO_DAYS,
    CRYPTO_SYMBOLS,
    TRADING_DAYS,
    calendar_days,
    periods_per_year,
)

NA = None  # JSON null == NOT AVAILABLE at the UI layer

PASS = "PASS"
FAIL = "FAIL"
CURRENT = "CURRENT"
NOT_RUN = "NOT_RUN"
NOT_AVAILABLE = "NOT_AVAILABLE"

# The scientific pipeline an experiment flows through, in order.
PIPELINE = [
    ("hypothesis", "HYPOTHESIS"),
    ("data", "DATA"),
    ("train", "TRAIN"),
    ("research_gate", "RESEARCH GATE"),
    ("replication", "REPLICATION"),
    ("walk_forward", "WALK-FORWARD"),
    ("adversarial", "ADVERSARIAL"),
    ("ablation", "ABLATION"),
    ("dsr", "DSR"),
    ("fdr", "FDR"),
    ("validation", "VALIDATION"),
    ("holdout", "HOLDOUT"),
    ("champion", "CHAMPION"),
]

# Validation evidence surfaced per candidate. Never collapsed into one score.
VALIDATION_TESTS = [
    ("significance", "Statistical significance"),
    ("p_value", "p-value"),
    ("n_effective", "Effective sample size"),
    ("replication", "Replication"),
    ("walk_forward", "Walk-forward"),
    ("adversarial", "Adversarial"),
    ("ablation", "Ablation"),
    ("dsr", "DSR"),
    ("fdr", "FDR"),
    ("validation", "Validation gate"),
    ("holdout", "Hidden holdout"),
    ("cross_market", "Cross-market"),
    ("paper_trading", "Paper trading"),
]

# 5 bps per position change — the default cost model evaluation.evaluate applies.
COST_PER_SIDE = 0.0005


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    status: str
    detail: str | None = None

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "status": self.status, "detail": self.detail}


def clean(value):
    """NaN/Inf are not data — they become NOT AVAILABLE rather than junk pixels."""
    if isinstance(value, float) and not math.isfinite(value):
        return NA
    return value


class Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return super().default(o)


def dumps(payload) -> bytes:
    return json.dumps(payload, cls=Encoder, allow_nan=False).encode()
