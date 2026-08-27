"""Dashboard computations and payload assembly.

Every number returned by this module is derived from a Quant Loop artifact or
database row. Where a value cannot be derived it is ``None`` (rendered N/A).
The return math mirrors ``quant_loop_trader.evaluation.evaluate`` exactly:
non-overlapping h-day buckets, long-or-flat positions, 5 bps per position
change. Tests assert the reproduction against sealed metrics.json values.
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone

from quant_loop_trader.dashboard import queries as q
from quant_loop_trader.dashboard.schemas import (
    COST_PER_SIDE,
    FAIL,
    NA,
    NOT_AVAILABLE,
    NOT_RUN,
    PASS,
    Stage,
    calendar_days,
    clean,
    periods_per_year,
)

# --- return / risk primitives ----------------------------------------------


def bucket_returns(rows: list[dict], horizon: int,
                   cost: float = COST_PER_SIDE,
                   entry_cost: bool = True) -> dict:
    """Non-overlapping h-day buckets — the engine's own return construction.

    rows: prediction records (event_time, close, y_pred), sorted by event_time.
    """
    h = max(1, int(horizon))
    prices = [float(r["close"]) for r in rows]
    dates = [r["event_time"] for r in rows]
    preds = [int(r["y_pred"]) for r in rows]
    n_buckets = min(len(preds), max(0, (len(prices) - 1) // h))
    if len(prices) < 2 or n_buckets <= 0:
        return {"dates": [], "start_date": dates[0] if dates else None,
                "bench": [], "gross": [], "net": [], "positions": [],
                "turnover": None, "n_buckets": 0}

    ends = list(range(h, h * n_buckets + 1, h))
    bench = [prices[e] / prices[e - h] - 1 for e in ends]
    pos = [preds[(i * h)] for i in range(n_buckets)]
    gross = [b * p for b, p in zip(bench, pos)]
    changes = [abs(pos[i + 1] - pos[i]) for i in range(n_buckets - 1)]
    net = list(gross)
    for i, ch in enumerate(changes):
        net[i + 1] -= ch * cost
    if entry_cost and pos and pos[0] > 0:
        net[0] -= cost
    return {
        "dates": [dates[e] for e in ends],
        "start_date": dates[0],
        "bench": bench,
        "gross": gross,
        "net": net,
        "positions": pos,
        "turnover": (sum(changes) / len(changes)) if changes else 0.0,
        "n_buckets": n_buckets,
    }


def compound(returns: list[float]) -> list[float]:
    """Wealth path from period returns, starting at 1.0 (no smoothing)."""
    out, w = [], 1.0
    for r in returns:
        w *= (1 + r)
        out.append(w)
    return out


def drawdown_series(equity: list[float]) -> list[float]:
    """equity / running_max(equity) - 1."""
    out, peak = [], -math.inf
    for v in equity:
        peak = max(peak, v)
        out.append(v / peak - 1 if peak > 0 else 0.0)
    return out


def max_drawdown(equity: list[float]) -> float:
    """Prefixed with initial capital 1.0, exactly like evaluation._max_drawdown."""
    if not equity:
        return 0.0
    return min(drawdown_series([1.0] + equity))


def sharpe(returns: list[float], ppy: float) -> float | None:
    if len(returns) < 2:
        return None
    sd = statistics.pstdev(returns)
    if sd == 0:
        return 0.0
    return math.sqrt(ppy) * statistics.fmean(returns) / sd


def downside_deviation(returns: list[float]) -> float:
    if not returns:
        return 0.0
    return math.sqrt(statistics.fmean([min(r, 0.0) ** 2 for r in returns]))


def rolling(values: list[float], window: int, fn) -> list[float | None]:
    """Trailing window; positions with insufficient observations stay None."""
    out: list[float | None] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(None)
        else:
            out.append(fn(values[i + 1 - window:i + 1]))
    return out


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = p * (len(s) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def correlation(a: list[float], b: list[float]) -> float | None:
    n = min(len(a), len(b))
    if n < 3:
        return None
    a, b = a[:n], b[:n]
    sa, sb = statistics.pstdev(a), statistics.pstdev(b)
    if sa == 0 or sb == 0:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (n * sa * sb)


def realized_vol(returns: list[float], days_per_year: int) -> float | None:
    if len(returns) < 2:
        return None
    return statistics.pstdev(returns) * math.sqrt(days_per_year)


# --- experiment views -------------------------------------------------------


def _cfg(art: dict) -> dict:
    return art.get("config") or (art.get("report") or {}).get("config") or {}


def _metrics(art: dict, variant: str = "improved") -> dict:
    m = art.get("metrics") or {}
    if variant in m:
        return m[variant] or {}
    rep = art.get("report") or {}
    return rep.get(f"{variant}_metrics") or {}


def _reviews(art: dict) -> dict:
    val = art.get("validation") or {}
    return {r["reviewer"]: r for r in val.get("reviews", [])}


def _issue_present(art: dict, *prefixes: str) -> bool:
    issues = (art.get("validation") or {}).get("issues_found") or []
    return any(i.startswith(p) or p in i for i in issues for p in prefixes)


# Sealed metrics are authoritative. The chart is a reconstruction from the
# prediction bundle, and the engine's return convention has changed over time
# (daily then h-day buckets; costs added; first-bucket entry cost added). Pick
# the convention that reproduces this bundle's own sealed cumulative return, and
# say so when none of them does rather than drawing a curve that disagrees.
CONVENTIONS = (
    ("h-day buckets, 5bps per side incl. entry", None, COST_PER_SIDE, True),
    ("h-day buckets, 5bps per side excl. entry", None, COST_PER_SIDE, False),
    ("h-day buckets, no costs", None, 0.0, True),
    ("daily buckets, no costs", 1, 0.0, True),
)


def _reconcile(rows: list[dict], horizon: int, sealed: dict) -> tuple[dict, tuple, bool]:
    """(bucket_returns result, (label, bucket_days, cost, entry), reconciled)."""
    target = sealed.get("cumulative_return_strategy")
    first = None
    for label, days, cost, entry in CONVENTIONS:
        days = horizon if days is None else days
        b = bucket_returns(rows, days, cost, entry)
        conv = (label, days, cost, entry)
        if first is None:
            first = (b, conv)
        if target is None or not b["n_buckets"]:
            continue
        got = compound(b["net"])[-1] - 1
        if abs(got - target) <= 1e-9 * max(1.0, abs(target)):
            return b, conv, True
    b, conv = first
    return b, conv, target is None or not b["n_buckets"]


def curve(experiment_id: str, variant: str = "improved") -> dict:
    """Equity, benchmark and drawdown series for one experiment, from predictions."""
    art = q.artifacts(experiment_id)
    cfg = _cfg(art)
    horizon = int(cfg.get("horizon") or 1)
    ticker = cfg.get("ticker") or "SPY"
    try:
        rows = q.predictions(experiment_id, variant)
    except q.DataUnavailable as exc:
        return {"experiment_id": experiment_id, "available": False,
                "reason": str(exc), "points": []}

    b, conv, reconciled = _reconcile(rows, horizon, _metrics(art, variant))
    bucket_days = conv[1]
    if not b["n_buckets"]:
        return {"experiment_id": experiment_id, "available": False,
                "reason": "insufficient_observations", "points": [],
                "n_buckets": 0}

    strat = compound(b["net"])
    gross = compound(b["gross"])
    bench = compound(b["bench"])
    dd = drawdown_series([1.0] + strat)[1:]
    ppy = periods_per_year(ticker, bucket_days)
    points = [
        {
            "t": str(d),
            "strategy": s,
            "strategy_gross": g,
            "benchmark": bm,
            "drawdown": ddv,
            "return": r,
            "benchmark_return": br,
            "position": p,
        }
        for d, s, g, bm, ddv, r, br, p in zip(
            b["dates"], strat, gross, bench, dd, b["net"], b["bench"], b["positions"]
        )
    ]
    return {
        "experiment_id": experiment_id,
        "variant": variant,
        "available": True,
        "ticker": ticker,
        "horizon": horizon,
        "bucket_days": bucket_days,
        "pipeline_version": int(cfg.get("pipeline_version") or 1),
        "return_convention": conv[0],
        "reconciled": reconciled,
        "periods_per_year": ppy,
        "calendar_days": calendar_days(ticker),
        "start": str(b["start_date"]),
        "points": points,
        "n_buckets": b["n_buckets"],
        "cost_per_side_bps": conv[2] * 1e4,
        "source": f"data/experiments/{experiment_id}/predictions_{variant}.parquet",
    }


def performance(experiment_id: str, variant: str = "improved") -> dict:
    """Headline performance. Sealed metrics are authoritative; series-derived
    extras (CAGR, annualized vol, excess) are recomputed from the same source."""
    art = q.artifacts(experiment_id)
    cfg = _cfg(art)
    sealed = _metrics(art, variant)
    c = curve(experiment_id, variant)
    ticker = cfg.get("ticker") or "SPY"
    horizon = int(cfg.get("horizon") or 1)
    ppy = c.get("periods_per_year") or periods_per_year(ticker, horizon)

    out = {
        "experiment_id": experiment_id,
        "variant": variant,
        "ticker": ticker,
        "horizon": horizon,
        "periods_per_year": ppy,
        "calendar_days": calendar_days(ticker),
        "net_return": clean(sealed.get("cumulative_return_strategy")),
        "gross_return_sum": clean(sealed.get("gross_return")),
        "benchmark_return": clean(sealed.get("cumulative_return_benchmark")),
        "sharpe": clean(sealed.get("sharpe_strategy")),
        "sharpe_benchmark": clean(sealed.get("sharpe_benchmark")),
        "sortino": clean(sealed.get("sortino_ratio")),
        "calmar": clean(sealed.get("calmar_ratio")),
        "max_drawdown": clean(sealed.get("max_drawdown_strategy")),
        "max_drawdown_benchmark": clean(sealed.get("max_drawdown_benchmark")),
        "volatility_per_bucket": clean(sealed.get("volatility_strategy")),
        "downside_deviation_per_bucket": clean(sealed.get("downside_deviation")),
        "win_rate": clean(sealed.get("win_rate")),
        "profit_factor": clean(sealed.get("profit_factor")),
        "expectancy": clean(sealed.get("expectancy")),
        "n_trades": clean(sealed.get("n_trades")),
        "turnover": clean(sealed.get("turnover")),
        "n_return_buckets": clean(sealed.get("n_return_buckets")),
        "n_test": clean(sealed.get("n_test")),
        "accuracy": clean(sealed.get("accuracy")),
        "precision": clean(sealed.get("precision")),
        "recall": clean(sealed.get("recall")),
        "brier_score": clean(sealed.get("brier_score")),
        "p_value": clean(sealed.get("stat_pvalue")),
        "return_ci95": sealed.get("return_ci95"),
        "var_95": clean(sealed.get("var_95")),
        "expected_shortfall_95": clean(sealed.get("expected_shortfall_95")),
        "transaction_cost_adj_return": clean(sealed.get("transaction_cost_adj_return")),
    }
    if out["net_return"] is not None and out["benchmark_return"] is not None:
        out["excess_return"] = out["net_return"] - out["benchmark_return"]
    else:
        out["excess_return"] = None
    if out["gross_return_sum"] is not None and out["transaction_cost_adj_return"] is not None:
        out["cost_drag"] = out["transaction_cost_adj_return"] - out["gross_return_sum"]
    else:
        out["cost_drag"] = None

    if c.get("available"):
        nets = [p["return"] for p in c["points"]]
        benches = [p["benchmark_return"] for p in c["points"]]
        n = len(nets)
        final = c["points"][-1]["strategy"]
        out.update({
            "cagr": (final ** (ppy / n) - 1) if final > 0 and n else None,
            "benchmark_cagr": (
                c["points"][-1]["benchmark"] ** (ppy / n) - 1
                if c["points"][-1]["benchmark"] > 0 and n else None
            ),
            "annualized_volatility": statistics.pstdev(nets) * math.sqrt(ppy) if n > 1 else None,
            "annualized_downside_volatility": downside_deviation(nets) * math.sqrt(ppy) if n else None,
            "benchmark_annualized_volatility": (
                statistics.pstdev(benches) * math.sqrt(ppy) if n > 1 else None
            ),
            "current_drawdown": c["points"][-1]["drawdown"],
            "exposure": statistics.fmean([1.0 if p["position"] else 0.0 for p in c["points"]]),
            "best_period": max(nets),
            "worst_period": min(nets),
            "largest_loss": min(nets) if min(nets) < 0 else 0.0,
        })
        out["return_over_vol"] = (
            out["cagr"] / out["annualized_volatility"]
            if out.get("cagr") is not None and out.get("annualized_volatility") else None
        )
        out["return_over_max_dd"] = (
            out["cagr"] / abs(out["max_drawdown"])
            if out.get("cagr") is not None and out.get("max_drawdown") else None
        )
    return out


def risk(experiment_id: str, variant: str = "improved") -> dict:
    """Drawdown, volatility and tail risk from the reconstructed equity curve."""
    c = curve(experiment_id, variant)
    if not c.get("available"):
        return {"experiment_id": experiment_id, "available": False,
                "reason": c.get("reason")}
    pts = c["points"]
    nets = [p["return"] for p in pts]
    dd = [p["drawdown"] for p in pts]
    ppy = c["periods_per_year"]
    h = c.get("bucket_days") or c["horizon"]
    cal = c["calendar_days"]

    trough = min(range(len(dd)), key=lambda i: dd[i])
    peak = max((i for i in range(trough + 1) if dd[i] == 0), default=0)
    recovery = next((i for i in range(trough, len(dd)) if dd[i] == 0), None)

    underwater, longest_uw, cur_uw, longest_rec = 0, 0, 0, 0
    for v in dd:
        if v < 0:
            underwater += 1
            longest_uw = max(longest_uw, underwater)
        else:
            longest_rec = max(longest_rec, underwater)
            underwater = 0
    cur_uw = underwater

    # Windows are stated in buckets AND days: with h-day buckets a "7d" window
    # cannot exist, so each entry reports the span it actually covers.
    windows = []
    for label, days in (("7D", 7), ("30D", 30), ("90D", 90), ("1Y", cal)):
        k = max(2, round(days / h))
        span = k * h
        windows.append({
            "label": label,
            "requested_days": days,
            "span_days": span,
            "buckets": k,
            "value": realized_vol(nets[-k:], ppy) if len(nets) >= k else None,
            "available": len(nets) >= k,
        })

    var95 = quantile(nets, 0.05)
    tail = [r for r in nets if var95 is not None and r <= var95]
    return {
        "experiment_id": experiment_id,
        "available": True,
        "horizon": h,
        "bucket_days": h,
        "calendar_days": cal,
        "periods_per_year": ppy,
        "annualized_volatility": realized_vol(nets, ppy),
        "annualized_downside_volatility": downside_deviation(nets) * math.sqrt(ppy) if nets else None,
        "realized_volatility": windows,
        "var_95": var95,
        "expected_shortfall_95": statistics.fmean(tail) if tail else None,
        "largest_loss": min(nets) if nets else None,
        "worst_period": {"t": pts[nets.index(min(nets))]["t"], "return": min(nets)} if nets else None,
        "current_drawdown": dd[-1],
        "max_drawdown": min(dd + [0.0]),
        "max_drawdown_period": {
            "peak": pts[peak]["t"],
            "trough": pts[trough]["t"],
            "recovered": pts[recovery]["t"] if recovery is not None else None,
        },
        "periods_underwater_current": cur_uw,
        "periods_underwater_longest": longest_uw,
        "longest_recovery_periods": longest_rec,
        "days_underwater_current": cur_uw * h,
        "days_underwater_longest": longest_uw * h,
        "longest_recovery_days": longest_rec * h,
        "note": f"one period = {h} trading days (non-overlapping horizon bucket)",
    }


ROLLING_WINDOWS = (30, 90, 180)


def rolling_performance(experiment_id: str, variant: str = "improved") -> dict:
    """Rolling Sharpe / return / volatility, and the edge-decay comparison."""
    c = curve(experiment_id, variant)
    if not c.get("available"):
        return {"experiment_id": experiment_id, "available": False,
                "reason": c.get("reason"), "windows": {}, "edge": _edge_unavailable()}
    pts = c["points"]
    nets = [p["return"] for p in pts]
    ppy = c["periods_per_year"]
    h = c.get("bucket_days") or c["horizon"]
    dates = [p["t"] for p in pts]

    windows = {}
    for days in ROLLING_WINDOWS:
        k = max(2, round(days / h))
        if len(nets) < k:
            windows[f"{days}D"] = {"available": False,
                                   "reason": f"needs {k} buckets, have {len(nets)}"}
            continue
        windows[f"{days}D"] = {
            "available": True,
            "buckets": k,
            "t": dates,
            "sharpe": rolling(nets, k, lambda w: sharpe(w, ppy)),
            "return": rolling(nets, k, lambda w: math.prod(1 + r for r in w) - 1),
            "volatility": rolling(nets, k, lambda w: realized_vol(w, ppy)),
        }
    return {
        "experiment_id": experiment_id,
        "available": True,
        "horizon": h,
        "windows": windows,
        "edge": edge_decay(nets, [p["benchmark_return"] for p in pts], ppy, h),
    }


def _edge_unavailable(reason: str = "insufficient_observations") -> dict:
    return {"status": NOT_AVAILABLE, "reason": reason}


def edge_decay(nets: list[float], bench: list[float], ppy: float, h: int) -> dict:
    """Recent vs full-period Sharpe and excess return. No black-box score.

    Classification is a stated rule over two visible ratios, and both the recent
    and historical inputs are returned so the rule can be checked by eye.
    """
    k = max(2, round(90 / h))
    if len(nets) < 2 * k:
        return _edge_unavailable(
            f"needs {2 * k} buckets for a recent-vs-history split, have {len(nets)}"
        )
    recent, history = nets[-k:], nets[:-k]
    r_bench, h_bench = bench[-k:], bench[:-k]
    rs, hs = sharpe(recent, ppy), sharpe(history, ppy)
    r_ex = math.prod(1 + r for r in recent) - math.prod(1 + r for r in r_bench)
    h_ex = math.prod(1 + r for r in history) - math.prod(1 + r for r in h_bench)

    if hs is None or hs <= 0:
        status = NOT_AVAILABLE
        ratio = None
        reason = "historical Sharpe is not positive — no edge to decay from"
    else:
        ratio = rs / hs if rs is not None else None
        reason = None
        if ratio is None:
            status = NOT_AVAILABLE
        elif ratio >= 0.7:
            status = "STABLE"
        elif ratio >= 0.3:
            status = "WEAKENING"
        else:
            status = "SEVERE_DECAY"
    return {
        "status": status,
        "reason": reason,
        "window_buckets": k,
        "recent_sharpe": rs,
        "historical_sharpe": hs,
        "sharpe_ratio": ratio,
        "recent_excess_return": r_ex,
        "historical_excess_return": h_ex,
        "rule": "STABLE >= 0.70 x historical Sharpe; WEAKENING >= 0.30; else SEVERE_DECAY",
    }


# --- pipeline / validation --------------------------------------------------


def _registry_status(experiment_id: str, registry: dict[str, dict] | None = None) -> str | None:
    registry = registry if registry is not None else _registry_map()
    row = registry.get(f"{experiment_id}_improved")
    return row["status"] if row else None


def _registry_map(authoritative_only: bool = False) -> dict[str, dict]:
    """model_registry rows. `model_registry` carries no authoritative column of
    its own, so lifecycle counts join it back onto the authoritative experiments."""
    try:
        rows = {r["model_id"]: r for r in q.model_registry_rows()}
    except q.DataUnavailable:
        return {}
    if not authoritative_only:
        return rows
    flags = q.authoritative_ids()
    if flags is None:
        return rows
    good, quarantined = flags
    return {k: v for k, v in rows.items() if q.stem(k) not in quarantined}


def _authoritative_flag(eid: str, flags) -> bool | None:
    """True authoritative, False quarantined, None no database record yet."""
    if flags is None:
        return None
    good, quarantined = flags
    return True if eid in good else (False if eid in quarantined else None)


def authority_available() -> bool:
    """Whether the database can vouch for provenance at all."""
    return q.authoritative_ids() is not None


def authoritative(rows: list[dict] | None = None) -> list[dict]:
    """Rows that ARE current research evidence — `authoritative is True` only.

    Every statistic (funnel, hypothesis tallies, pass rates, lifecycle counts,
    research progress) is computed over this population. When the database is
    unreadable this is empty by construction; callers must check
    `authority_available()` and report N/A rather than a filesystem count.
    """
    rows = rows if rows is not None else experiment_index()
    return [r for r in rows if r["authoritative"] is True]


def visible(rows: list[dict] | None = None) -> list[dict]:
    """Rows the control room shows — authoritative plus not-yet-recorded runs.

    An in-flight experiment has no `experiments` row until it seals, so it is
    unknown, not quarantined. It belongs on the screen; it does not belong in
    any evidence statistic until the database says `authoritative`.
    """
    rows = rows if rows is not None else experiment_index()
    return [r for r in rows if r["authoritative"] is not False]


def population() -> dict:
    """How much of what is on disk is authoritative — never hide the rest silently."""
    rows = experiment_index()
    if q.authoritative_ids() is None:
        return {"basis": "UNKNOWN", "on_disk": len(rows), "authoritative": None,
                "quarantined": None, "unrecorded": None,
                "reason": "database unreadable — authoritative flags unavailable"}
    return {
        "basis": "AUTHORITATIVE",
        "on_disk": len(rows),
        "authoritative": sum(1 for r in rows if r["authoritative"] is True),
        "quarantined": sum(1 for r in rows if r["authoritative"] is False),
        "unrecorded": sum(1 for r in rows if r["authoritative"] is None),
        "reason": "quarantined runs predate the current pipeline and are excluded",
    }


def stages(experiment_id: str, art: dict | None = None,
           registry: dict | None = None) -> list[dict]:
    """Pipeline state of one experiment, read from its sealed artifacts."""
    art = art or q.artifacts(experiment_id)
    cfg = _cfg(art)
    rep = art.get("report") or {}
    val = art.get("validation") or {}
    hard = val.get("hardening") or {}
    rev = _reviews(art)
    files = art.get("files", {})
    running = art["exists"] and not files.get("report.json")

    def review(name: str) -> str:
        r = rev.get(name)
        if not r:
            return NOT_RUN
        return PASS if r["approval_status"] == "APPROVED" else FAIL

    out: list[Stage] = []
    out.append(Stage("hypothesis", "HYPOTHESIS",
                     PASS if rep.get("hypothesis") else ("CURRENT" if running else NOT_RUN),
                     (rep.get("hypothesis") or "")[:120] or None))
    dataset_id = cfg.get("dataset_id")
    snapshot = q.paths()["datasets"] / f"{dataset_id}.parquet" if dataset_id else None
    if snapshot and snapshot.exists():
        data_status, data_detail = PASS, dataset_id
    elif not dataset_id:
        data_status, data_detail = (NOT_RUN, None)
    else:
        # The experiment ran on a snapshot that is no longer on disk (experiments
        # predating sealed snapshots). Absent evidence — not a failed data stage.
        data_status = NOT_AVAILABLE
        data_detail = f"{dataset_id} — snapshot not on disk"
    out.append(Stage("data", "DATA", data_status, data_detail))
    out.append(Stage("train", "TRAIN",
                     PASS if files.get("predictions.lock") else ("CURRENT" if running else NOT_RUN),
                     cfg.get("model_version")))
    decision = rep.get("decision")
    out.append(Stage("research_gate", "RESEARCH GATE",
                     NOT_RUN if not decision else (FAIL if decision == "REJECT" else PASS),
                     decision))
    out.append(Stage("replication", "REPLICATION", review("independent_replicator")))

    wf = hard.get("walk_forward")
    out.append(Stage(
        "walk_forward", "WALK-FORWARD",
        NOT_RUN if not wf else (PASS if wf.get("stable_across_time") else FAIL),
        None if not wf else f"mean acc {wf.get('mean_accuracy')} over {len(wf.get('folds', []))} folds",
    ))
    out.append(Stage("adversarial", "ADVERSARIAL", review("adversarial_reviewer")))

    abl = hard.get("ablation")
    out.append(Stage(
        "ablation", "ABLATION",
        NOT_RUN if abl is None else (FAIL if _issue_present(art, "ablation:") else PASS),
        None if abl is None else f"{len(abl)} variants",
    ))

    mt = hard.get("multiple_testing")
    out.append(Stage(
        "dsr", "DSR",
        NOT_RUN if not mt else (PASS if mt.get("verdict") not in ("PROBABLY_LUCK", "LOW_CONFIDENCE") else FAIL),
        None if not mt else f"DSR {mt.get('dsr')} · {mt.get('verdict')} · {mt.get('n_trials')} trials",
    ))

    # The engine only runs Benjamini-Hochberg when the verified same-family
    # p-value set reaches 5; the artifact records the outcome only as an issue.
    if _issue_present(art, "multiple_testing:fdr_not_significant"):
        fdr_status, fdr_detail = FAIL, "BH rejection failed at FDR 0.10"
    elif mt:
        fdr_status, fdr_detail = PASS, "no BH rejection raised at FDR 0.10"
    else:
        fdr_status, fdr_detail = NOT_RUN, None
    out.append(Stage("fdr", "FDR", fdr_status, fdr_detail))

    approval = val.get("approval_status")
    out.append(Stage("validation", "VALIDATION",
                     NOT_RUN if not approval else (PASS if approval == "APPROVED" else FAIL),
                     f"{len(val.get('issues_found') or [])} issues" if approval else None))

    ho = art.get("holdout_report")
    holdout_detail = holdout_summary(ho)
    out.append(Stage("holdout", "HOLDOUT",
                     NOT_RUN if ho is None else (PASS if ho.get("promoted") else FAIL),
                     holdout_detail))

    status = _registry_status(experiment_id, registry)
    out.append(Stage("champion", "CHAMPION",
                     PASS if status == "champion" else NOT_RUN,
                     f"registry: {status}" if status else None))
    return [s.to_dict() for s in out]


def holdout_summary(ho: dict | None) -> str | None:
    """One honest line about a holdout adjudication, from its own record."""
    if ho is None:
        return None
    if ho.get("reason"):
        return str(ho["reason"])[:160]
    if ho.get("holdout_accuracy") is not None:
        return (f"acc {ho['holdout_accuracy']:.4f} vs base rate "
                f"{ho.get('base_rate'):.4f} · n={ho.get('n_holdout')}")
    return "promoted" if ho.get("promoted") else "not promoted"


def current_stage(stage_list: list[dict]) -> str | None:
    """Where the experiment stands: the running stage, else where it stopped."""
    for s in stage_list:
        if s["status"] == "CURRENT":
            return s["label"]
    for s in stage_list:
        if s["status"] == FAIL:
            return s["label"]
    passed = [s for s in stage_list if s["status"] == PASS]
    return passed[-1]["label"] if passed else None


def validation_view(experiment_id: str, art: dict | None = None) -> dict:
    """Per-test evidence. Never collapsed into a single confidence score."""
    art = art or q.artifacts(experiment_id)
    val = art.get("validation") or {}
    rev = _reviews(art)
    hard = val.get("hardening") or {}
    m = _metrics(art, "improved")
    rep = art.get("report") or {}
    wf = hard.get("walk_forward") or {}
    mt = hard.get("multiple_testing") or {}
    ho = art.get("holdout_report")
    st = {s["key"]: s for s in stages(experiment_id, art)}

    def status(key):
        return st[key]["status"] if key in st else NOT_RUN

    return {
        "experiment_id": experiment_id,
        "approval_status": val.get("approval_status") or NOT_RUN,
        "issues": val.get("issues_found") or [],
        "tests": {
            "significance": {
                "status": (PASS if rev.get("statistical_reviewer", {}).get("approval_status") == "APPROVED"
                           else FAIL if "statistical_reviewer" in rev else NOT_RUN),
                "p_value": clean(rep.get("candidate_stat_pvalue", m.get("stat_pvalue"))),
                "n_effective": clean(m.get("n_return_buckets")),
                "n_test": clean(m.get("n_test")),
                "issues": rev.get("statistical_reviewer", {}).get("issues_found", []),
            },
            "replication": {
                "status": status("replication"),
                "issues": rev.get("independent_replicator", {}).get("issues_found", []),
                "tests": rev.get("independent_replicator", {}).get("tests_completed", []),
            },
            "adversarial": {
                "status": status("adversarial"),
                "issues": rev.get("adversarial_reviewer", {}).get("issues_found", []),
                "tests": rev.get("adversarial_reviewer", {}).get("tests_completed", []),
            },
            "walk_forward": {
                "status": status("walk_forward"),
                "mean_accuracy": clean(wf.get("mean_accuracy")),
                "dispersion": clean(wf.get("accuracy_dispersion")),
                "folds": wf.get("folds", []),
            },
            "ablation": {"status": status("ablation"), "variants": hard.get("ablation")},
            "dsr": {
                "status": status("dsr"),
                "dsr": clean(mt.get("dsr")),
                "verdict": mt.get("verdict"),
                "n_trials": mt.get("n_trials"),
            },
            "fdr": {"status": status("fdr"), "detail": st.get("fdr", {}).get("detail")},
            "holdout": {
                "status": status("holdout"),
                "promoted": None if ho is None else ho.get("promoted"),
                "reason": None if ho is None else (
                    ho.get("reason") or st.get("holdout", {}).get("detail")),
                "accuracy": None if ho is None else clean(ho.get("holdout_accuracy")),
                "base_rate": None if ho is None else clean(ho.get("base_rate")),
                "n_holdout": None if ho is None else ho.get("n_holdout"),
                "economic_gate": None if ho is None else ho.get("economic_gate"),
            },
            "cross_market": {"status": NOT_AVAILABLE,
                             "detail": "single-market research programme (SPY only)"},
            "paper_trading": {"status": NOT_RUN,
                              "detail": "paper trading disabled until QLT_PAPER_ENABLED=true"},
        },
    }


# --- experiment index -------------------------------------------------------


def _duration_seconds(art: dict) -> float | None:
    if not art.get("started_at") or not art.get("sealed_at"):
        return None
    a = datetime.fromisoformat(art["started_at"])
    b = datetime.fromisoformat(art["sealed_at"])
    d = (b - a).total_seconds()
    return d if d >= 0 else None


@q.ttl_cache
def experiment_index() -> list[dict]:
    """One dense row per experiment directory — the table's source."""
    registry = _registry_map()
    cycle_of = _experiment_cycle_map()
    flags = q.authoritative_ids()
    out = []
    for eid in q.experiment_ids():
        art = q.artifacts(eid)
        rep = art.get("report") or {}
        cfg = _cfg(art)
        m = _metrics(art, "improved")
        val = art.get("validation") or {}
        ho = art.get("holdout_report")
        hard = val.get("hardening") or {}
        mt = hard.get("multiple_testing") or {}
        st = stages(eid, art, registry)
        running = art["exists"] and not art.get("report")
        net = clean(m.get("cumulative_return_strategy"))
        bench = clean(m.get("cumulative_return_benchmark"))
        out.append({
            "id": eid,
            "authoritative": _authoritative_flag(eid, flags),
            "cycle": cycle_of.get(eid),
            "market": cfg.get("ticker") or rep.get("config", {}).get("ticker"),
            "hypothesis": rep.get("hypothesis"),
            "research_question": rep.get("research_question"),
            "feature_family": cfg.get("feature_version_improved") or rep.get("feature_version"),
            "horizon": cfg.get("horizon"),
            "model": cfg.get("model_version") or rep.get("model_version"),
            "stage": "EXECUTION" if running else current_stage(st),
            "status": "RUNNING" if running else (rep.get("decision") or "UNSEALED"),
            "registry_status": _registry_status(eid, registry),
            "net_return": net,
            "benchmark_return": bench,
            "excess_return": (net - bench) if net is not None and bench is not None else None,
            "sharpe": clean(m.get("sharpe_strategy")),
            "max_drawdown": clean(m.get("max_drawdown_strategy")),
            "volatility": clean(m.get("volatility_strategy")),
            "accuracy": clean(m.get("accuracy")),
            "p_value": clean(rep.get("candidate_stat_pvalue", m.get("stat_pvalue"))),
            "dsr": clean(mt.get("dsr")),
            "dsr_verdict": mt.get("verdict"),
            "fdr": next((s["status"] for s in st if s["key"] == "fdr"), None),
            "validation": val.get("approval_status") or NOT_RUN,
            "holdout": (NOT_RUN if ho is None else (PASS if ho.get("promoted") else FAIL)),
            "started": art.get("started_at"),
            "sealed": art.get("sealed_at"),
            "duration_s": _duration_seconds(art),
            "seed": cfg.get("seed"),
            "train_period": cfg.get("train_period"),
            "test_period": cfg.get("test_period"),
            "dataset_id": cfg.get("dataset_id"),
        })
    out.sort(key=lambda r: r["started"] or "", reverse=True)
    return out


def experiment_detail(experiment_id: str) -> dict:
    art = q.artifacts(experiment_id)
    if not art["exists"]:
        raise KeyError(f"experiment_missing:{experiment_id}")  # a lookup miss, not an outage
    rep = art.get("report") or {}
    cfg = _cfg(art)
    lock = art.get("predictions") or {}
    return {
        "id": experiment_id,
        "report": rep,
        "config": cfg,
        "lineage": {
            "parent": rep.get("parent_experiment_id"),
            "branch": rep.get("research_branch"),
            "mutation_reason": rep.get("mutation_reason"),
            "creator": rep.get("creator_agent"),
            "code_version": (rep.get("reproducibility") or {}).get("code_version"),
            "dataset_checksum": cfg.get("dataset_checksum"),
            "dataset_snapshot": cfg.get("dataset_snapshot"),
            "sealed_artifacts": sorted(k for k in lock if k != "locked_at"),
            "locked_at": lock.get("locked_at"),
        },
        "stages": stages(experiment_id, art),
        "validation": validation_view(experiment_id, art),
        "performance": {
            "improved": performance(experiment_id, "improved"),
            "baseline": performance(experiment_id, "baseline"),
        },
        "risk": risk(experiment_id, "improved"),
        "regime_performance": rep.get("improved_error_analysis") or rep.get("error_analysis"),
        "error_analysis": rep.get("error_analysis"),
        "started": art.get("started_at"),
        "sealed": art.get("sealed_at"),
        "duration_s": _duration_seconds(art),
    }


# --- cycles -----------------------------------------------------------------


RUN_FRESH_S = 1800        # a live run touches its directory well inside 30 min
RUN_STALE_S = 24 * 3600   # older than a day with no seal is not coming back


def _in_flight() -> list[dict]:
    """Directories with no sealed report.

    `run_experiment` creates the directory before it inserts any database row,
    so an unsealed directory is evidence of a started process, never proof of a
    live one. Age plus session freshness separates the three cases; none of them
    is reported as a worker count.
    """
    hb = q.heartbeat() or {}
    session_live = bool(hb.get("timestamp")) and (
        _elapsed(hb["timestamp"], None) or 0) < RUN_FRESH_S
    out = []
    for eid in q.experiment_ids():
        art = q.artifacts(eid)
        if art.get("report"):
            continue
        started = art.get("started_at")
        age = None
        if started:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds()
        if age is not None and age < RUN_FRESH_S:
            state = "RUNNING"
        elif age is None or age > RUN_STALE_S or not session_live:
            state = "ORPHANED"
        else:
            state = "STALE"
        out.append({"id": eid, "started_at": started, "age_s": age, "state": state})
    return out


@q.ttl_cache
def cycles() -> list[dict]:
    """A cycle is one autonomy session (`autonomy.run_session`).

    Sessions are the only batch Quant Loop actually runs; the dashboard reads
    their summaries from data/logs/session.log and adds no state of its own.
    """
    sessions = q.session_records()
    flags = q.authoritative_ids()
    out = []
    for i, s in enumerate(sessions, start=1):
        results = s.get("results") or []
        ids = [r["experiment_id"] for r in results]
        sealed = [eid for eid in ids if (q.paths()["experiments"] / eid / "report.json").exists()]
        decisions = [r.get("decision") for r in results]
        validations = [r.get("validation_status") for r in results]
        started = s.get("session_started")
        finished = s.get("session_finished")
        out.append({
            "cycle_id": f"{started}",
            "cycle_number": i,
            "started_at": started,
            "completed_at": finished,
            "status": "COMPLETED" if finished else "UNKNOWN",
            "mode": s.get("mode"),
            "planned_experiments": s.get("budget"),
            "completed_experiments": s.get("executed"),
            "failed_experiments": len(ids) - len(sealed),
            "unused_budget": (s.get("budget") or 0) - (s.get("executed") or 0),
            "experiments": ids,
            "decisions": decisions,
            "keeps": sum(1 for d in decisions if d == "KEEP"),
            "improves": sum(1 for d in decisions if d == "IMPROVE"),
            "rejects": sum(1 for d in decisions if d == "REJECT"),
            "validation_passes": sum(1 for v in validations if v == "APPROVED"),
            "validation_failures": sum(1 for v in validations if v == "REJECTED"),
            # scheduler progress; `grid_remaining` counts a config as explored
            # even when its run was later quarantined
            "grid_remaining": s.get("grid_remaining"),
            "grid_remaining_basis": "scheduler",
            "authoritative_experiments": (None if flags is None
                                          else sum(1 for eid in ids if eid in flags[0])),
            "elapsed_s": _elapsed(started, finished),
            "source": "data/logs/session.log",
        })
    return out


def _elapsed(a: str | None, b: str | None) -> float | None:
    if not a:
        return None
    end = datetime.fromisoformat(b) if b else datetime.now(timezone.utc)
    return (end - datetime.fromisoformat(a)).total_seconds()


def _experiment_cycle_map() -> dict[str, int]:
    m = {}
    for c in cycles():
        for eid in c["experiments"]:
            m[eid] = c["cycle_number"]
    return m


def current_cycle() -> dict:
    """The live cycle if a run is in flight, otherwise the most recent session."""
    done = cycles()
    live = _in_flight()
    hb = q.heartbeat() or {}
    registry = _registry_map()

    if live:
        live.sort(key=lambda r: r["started_at"] or "")
        active = live[-1]
        art = q.artifacts(active["id"])
        cur_stages = stages(active["id"], art, registry)
        return {
            "cycle_number": len(done) + 1,
            "cycle_id": active["started_at"],
            "status": active["state"],
            "started_at": active["started_at"],
            "completed_at": None,
            "elapsed_s": active["age_s"],
            "planned_experiments": None,
            "completed_experiments": _sealed_since(done[-1]["completed_at"] if done else None),
            "failed_experiments": sum(1 for r in live if r["state"] == "STALLED"),
            "progress": None,
            "active_experiment": active["id"],
            "active_stage": "EXECUTION",
            "active_hypothesis": None,
            "active_market": _ticker_from_id(active["id"]),
            "active_horizon": _horizon_from_id(active["id"]),
            "stages": cur_stages,
            "next_experiment": None,
            "grid_remaining": hb.get("grid_remaining"),
            "note": "live cycle — the running experiment has not sealed its report yet",
        }

    if not done:
        return {"cycle_number": None, "status": "NO_DATA", "stages": [],
                "note": "no autonomy session has been recorded in data/logs/session.log"}

    c = dict(done[-1])
    last_id = c["experiments"][-1] if c["experiments"] else None
    art = q.artifacts(last_id) if last_id else {}
    rep = (art.get("report") or {}) if art else {}
    cfg = _cfg(art) if art else {}
    c.update({
        "status": "IDLE",
        "progress": (c["completed_experiments"] / c["planned_experiments"])
        if c.get("planned_experiments") else None,
        "active_experiment": None,
        "last_experiment": last_id,
        "active_stage": current_stage(stages(last_id, art, registry)) if last_id else None,
        "active_hypothesis": rep.get("hypothesis"),
        "active_market": cfg.get("ticker"),
        "active_horizon": cfg.get("horizon"),
        "stages": stages(last_id, art, registry) if last_id else [],
        "grid_remaining": hb.get("grid_remaining"),
        "next_experiment": None,
        "note": "no experiment in flight — last completed session shown",
    })
    return c


def _sealed_since(iso: str | None) -> int:
    if not iso:
        return 0
    return sum(1 for r in visible()
               if r["sealed"] and r["sealed"] > iso)


def _ticker_from_id(eid: str) -> str | None:
    parts = eid.split("_")
    return parts[1] if len(parts) > 2 else None


def _horizon_from_id(eid: str) -> int | None:
    parts = eid.split("_")
    if len(parts) > 2 and parts[2].endswith("d") and parts[2][:-1].isdigit():
        return int(parts[2][:-1])
    return None


# --- programme-level views --------------------------------------------------


def funnel() -> dict:
    pop = population()
    if pop["basis"] == "UNKNOWN":
        return {
            "population": pop, "available": False,
            "hypotheses": NA, "experiments": NA, "research_pass": NA,
            "validation_pass": NA, "holdout_pass": NA, "champions": NA,
            "validation_attempts": NA, "holdout_attempts": NA,
            "source": "unavailable — database cannot vouch for provenance",
        }
    rows = authoritative()
    sealed = [r for r in rows if r["status"] != "RUNNING" and r["status"] != "UNSEALED"]
    hypotheses = {r["hypothesis"] for r in rows if r["hypothesis"]}
    registry = _registry_map(authoritative_only=True)
    champions = sum(1 for r in registry.values() if r["status"] == "champion")
    return {
        "population": pop,
        "available": True,
        "hypotheses": len(hypotheses),
        "experiments": len(sealed),
        "research_pass": sum(1 for r in sealed if r["status"] in ("KEEP", "IMPROVE")),
        "validation_pass": sum(1 for r in sealed if r["validation"] == "APPROVED"),
        "holdout_pass": sum(1 for r in sealed if r["holdout"] == PASS),
        "champions": champions,
        "validation_attempts": sum(1 for r in sealed if r["validation"] != NOT_RUN),
        "holdout_attempts": sum(1 for r in sealed if r["holdout"] != NOT_RUN),
        "source": "authoritative experiment artifacts + model_registry",
    }


REJECTION_LABELS = [
    ("not_significant_vs_base_rate", "Failed significance"),
    ("not_significant_vs_coinflip", "Failed significance (legacy coin-flip test)"),
    ("label_randomisation", "Failed adversarial: label randomisation"),
    ("feature_shuffle", "Failed adversarial: feature shuffle"),
    ("regime_concentration", "Adversarial: regime concentration"),
    ("near_degenerate_minority_rate", "Degenerate predictions"),
    ("degenerate_constant_predictions", "Degenerate predictions"),
    ("sample_size_too_small", "Insufficient sample size"),
    ("walk_forward", "Failed walk-forward"),
    ("multiple_testing:deflated_sharpe_probably_luck", "DSR failure"),
    ("multiple_testing:dsr_low_confidence", "DSR low confidence"),
    ("multiple_testing:fdr_not_significant", "FDR failure"),
    ("ablation", "Failed ablation"),
    ("replication", "Failed replication"),
    ("metric_mismatch", "Failed replication: metric mismatch"),
    ("artifact_tampered", "Data integrity failure"),
    ("locked_artifact_missing", "Data integrity failure"),
    ("bundle_integrity", "Data integrity failure"),
    ("hardening_error", "Hardening error"),
]


def rejections() -> dict:
    """Why experiments fail — counted from validation issues and gate decisions."""
    counts: dict[str, int] = {}
    experiments_with_issues = 0
    gate_counts = {"No accuracy lift (REJECT)": 0, "Accuracy lift, Sharpe degraded (IMPROVE)": 0}
    holdout_fail = 0
    rows = [r["id"] for r in authoritative()]
    for eid in rows:
        art = q.artifacts(eid)
        rep = art.get("report") or {}
        if rep.get("decision") == "REJECT":
            gate_counts["No accuracy lift (REJECT)"] += 1
        elif rep.get("decision") == "IMPROVE":
            gate_counts["Accuracy lift, Sharpe degraded (IMPROVE)"] += 1
        issues = (art.get("validation") or {}).get("issues_found") or []
        if issues:
            experiments_with_issues += 1
        for issue in issues:
            label = next((lbl for key, lbl in REJECTION_LABELS if key in issue),
                         f"Other: {issue.split(':')[0]}")
            counts[label] = counts.get(label, 0) + 1
        ho = art.get("holdout_report")
        if ho is not None and not ho.get("promoted"):
            holdout_fail += 1
    total = sum(counts.values())
    items = [
        {"reason": k, "count": v, "pct": (v / total if total else None)}
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    return {
        "validation_issues": items,
        "total_issues": total,
        "experiments_with_issues": experiments_with_issues,
        "research_gate": [{"reason": k, "count": v} for k, v in gate_counts.items()],
        "holdout_failures": holdout_fail,
    }


def hypotheses() -> list[dict]:
    """Hypotheses actually researched, with their evidence trail."""
    rows = authoritative()
    memory = []
    try:
        memory = q.research_memory_rows()
    except q.DataUnavailable:
        pass
    by_hyp: dict[str, dict] = {}
    for r in rows:
        h = r["hypothesis"]
        if not h:
            continue
        e = by_hyp.setdefault(h, {
            "hypothesis": h, "research_question": r["research_question"],
            "experiments": 0, "keep": 0, "improve": 0, "reject": 0,
            "validation_pass": 0, "holdout_pass": 0, "markets": set(),
            "horizons": set(), "first_seen": r["started"], "last_seen": r["started"],
        })
        e["experiments"] += 1
        e["keep"] += r["status"] == "KEEP"
        e["improve"] += r["status"] == "IMPROVE"
        e["reject"] += r["status"] == "REJECT"
        e["validation_pass"] += r["validation"] == "APPROVED"
        e["holdout_pass"] += r["holdout"] == PASS
        e["markets"].add(r["market"])
        e["horizons"].add(r["horizon"])
        e["first_seen"] = min(e["first_seen"] or "", r["started"] or "") or None
        e["last_seen"] = max(e["last_seen"] or "", r["started"] or "") or None
    confidences = {}
    for m in memory:
        if m["hypothesis"] and m["hypothesis"] not in confidences:
            confidences[m["hypothesis"]] = m["confidence"]
    out = []
    for h, e in by_hyp.items():
        e["markets"] = sorted(x for x in e["markets"] if x)
        e["horizons"] = sorted(x for x in e["horizons"] if x)
        e["belief_confidence"] = confidences.get(h)
        e["status"] = ("REJECTED" if e["reject"] == e["experiments"]
                       else "UNDER_TEST")
        out.append(e)
    out.sort(key=lambda e: -e["experiments"])
    return out


def champions() -> dict:
    """Champion leaderboard from the model registry lifecycle, plus correlation."""
    registry = _registry_map(authoritative_only=True)
    lifecycle = {"champion": [], "eligible": [], "candidate": [], "rejected": []}
    for model_id, row in registry.items():
        if not model_id.endswith("_improved"):
            continue
        lifecycle.setdefault(row["status"], []).append(model_id[: -len("_improved")])

    def entry(eid: str, status: str) -> dict:
        perf = performance(eid)
        roll = rolling_performance(eid)
        return {
            "experiment_id": eid,
            "lifecycle": status.upper(),
            "market": perf.get("ticker"),
            "horizon": perf.get("horizon"),
            "net_return": perf.get("net_return"),
            "benchmark_return": perf.get("benchmark_return"),
            "excess_return": perf.get("excess_return"),
            "sharpe": perf.get("sharpe"),
            "sortino": perf.get("sortino"),
            "calmar": perf.get("calmar"),
            "max_drawdown": perf.get("max_drawdown"),
            "annualized_volatility": perf.get("annualized_volatility"),
            "p_value": perf.get("p_value"),
            "metrics_basis": "research test window (sealed metrics.json)",
            "holdout": _holdout_economics(eid),
            "edge": roll.get("edge"),
            "evidence": {
                s["key"]: s["status"] for s in stages(eid)
                if s["key"] in ("replication", "walk_forward", "adversarial",
                                "ablation", "dsr", "fdr", "validation", "holdout")
            },
        }

    champs = [entry(e, "champion") for e in lifecycle.get("champion", [])]
    eligible = [entry(e, "eligible") for e in lifecycle.get("eligible", [])]
    corr = correlation_matrix([c["experiment_id"] for c in champs])
    return {
        "champions": sorted(champs, key=lambda c: (c["sharpe"] is None, -(c["sharpe"] or 0))),
        "eligible": eligible,
        "counts": {k: len(v) for k, v in lifecycle.items()},
        "correlation": corr,
        "note": ("no model has passed validation + hidden holdout, so the registry "
                 "holds no champion" if not champs else None),
    }


def _holdout_economics(experiment_id: str) -> dict:
    """Hidden-holdout economics, straight from holdout_report.json.

    Quant Loop persists only the promotion verdict, accuracy, base rate, sample
    size and the economic gate; holdout drawdown/volatility/Sortino are computed
    during adjudication but never written, so they are reported unavailable
    rather than back-filled from research-window metrics.
    """
    ho = q.artifacts(experiment_id).get("holdout_report")
    if ho is None:
        return {"status": NOT_RUN, "available": False}
    gate = ho.get("economic_gate") or {}
    full = ho.get("holdout_metrics") or {}   # written since 006; absent on older reports
    ppy = periods_per_year(_ticker_from_id(experiment_id) or "SPY",
                           (_cfg(q.artifacts(experiment_id)) or {}).get("horizon") or 5)
    vol = clean(full.get("volatility_strategy"))
    out = {
        "status": PASS if ho.get("promoted") else FAIL,
        "available": True,
        "accuracy": clean(ho.get("holdout_accuracy")),
        "base_rate": clean(ho.get("base_rate")),
        "n": ho.get("n_holdout"),
        "net_return": clean(gate.get("compounded_net_return")),
        "sharpe": clean(gate.get("sharpe_strategy")),
        "benchmark_sharpe": clean(gate.get("sharpe_benchmark")),
        "max_drawdown": clean(full.get("max_drawdown_strategy")),
        "volatility_per_bucket": vol,
        "annualized_volatility": (vol * math.sqrt(ppy)) if vol is not None else NA,
        "sortino": clean(full.get("sortino_ratio")),
        "calmar": clean(full.get("calmar_ratio")),
        "var_95": clean(full.get("var_95")),
        "expected_shortfall": clean(full.get("expected_shortfall_95")),
        "turnover": clean(full.get("turnover")),
        "win_rate": clean(full.get("win_rate")),
        "metrics_source": ("holdout_report.holdout_metrics" if full
                           else "holdout_report.economic_gate"),
        "reason": holdout_summary(ho),
    }
    out["unavailable_fields"] = [k for k, v in out.items() if v is None
                                 and k not in ("reason", "status")]
    out["unavailable_reason"] = (None if full else
                                 "this report predates holdout_metrics persistence")
    return out


def correlation_matrix(ids: list[str]) -> dict:
    if len(ids) < 2:
        return {"available": False,
                "reason": f"needs 2+ champions, have {len(ids)}", "ids": ids}
    series = {}
    for eid in ids:
        c = curve(eid)
        if c.get("available"):
            series[eid] = {p["t"]: p["return"] for p in c["points"]}
    ids = [i for i in ids if i in series]
    matrix = []
    for a in ids:
        row = []
        for b in ids:
            shared = sorted(set(series[a]) & set(series[b]))
            row.append(correlation([series[a][t] for t in shared],
                                   [series[b][t] for t in shared]))
        matrix.append(row)
    flagged = [
        [ids[i], ids[j], matrix[i][j]]
        for i in range(len(ids)) for j in range(i + 1, len(ids))
        if matrix[i][j] is not None and matrix[i][j] > 0.8
    ]
    return {"available": True, "ids": ids, "matrix": matrix,
            "highly_correlated": flagged,
            "note": "return correlation on aligned horizon buckets"}


# --- market -----------------------------------------------------------------


def market(ticker: str = "SPY") -> dict:
    """Market state computed from the research price snapshot — no live feed."""
    try:
        rows = q.price_history(ticker)
    except q.DataUnavailable as exc:
        return {"ticker": ticker, "available": False, "reason": str(exc)}
    if len(rows) < 30:
        return {"ticker": ticker, "available": False, "reason": "insufficient_history"}

    closes = [float(r["close"]) for r in rows]
    dates = [str(r["event_time"]) for r in rows]
    vols = [int(r["volume"]) for r in rows if r.get("volume") is not None]
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    cal = calendar_days(ticker)

    def rv(n):
        return realized_vol(rets[-n:], cal) if len(rets) >= n else None

    rv30 = rv(30)
    hist = [x for x in rolling(rets, 30, lambda w: realized_vol(w, cal)) if x is not None]
    pct = (sum(1 for x in hist if x <= rv30) / len(hist)) if hist and rv30 else None
    regime = None
    if pct is not None:
        regime = ("LOW" if pct < 0.2 else "NORMAL" if pct < 0.6
                  else "HIGH" if pct < 0.9 else "EXTREME")
    sma = lambda n: statistics.fmean(closes[-n:]) if len(closes) >= n else None
    return {
        "ticker": ticker,
        "available": True,
        "as_of": dates[-1],
        "close": closes[-1],
        "volume": vols[-1] if vols else None,
        "change_1d": closes[-1] / closes[-2] - 1,
        "change_30d": closes[-1] / closes[-31] - 1 if len(closes) > 31 else None,
        "sma50": sma(50),
        "sma200": sma(200),
        "trend": (None if not sma(200) else
                  "UP" if closes[-1] > sma(50) > sma(200) else
                  "DOWN" if closes[-1] < sma(50) < sma(200) else "MIXED"),
        "realized_volatility": {"7d": rv(7), "30d": rv30, "90d": rv(90),
                                f"{cal}d": rv(cal)},
        "volatility_regime": regime,
        "volatility_percentile": pct,
        "calendar_days": cal,
        "history": [{"t": t, "close": c} for t, c in zip(dates, closes)],
        "rows": len(rows),
        "source": f"data/processed/{ticker}.parquet",
        "unavailable_fields": ["funding", "open_interest", "basis", "liquidations",
                               "spot_perp_ratio"],
        "unavailable_reason": "no derivatives/crypto connector implemented in Quant Loop",
    }


# --- system / activity ------------------------------------------------------

HEARTBEAT_STALE_S = 48 * 3600


@q.ttl_cache
def _lifecycle_inconsistencies() -> list[dict]:
    """Contradictions between the registry and the evidence on disk.

    Neither side is corrected here — the dashboard is read-only — but a
    half-committed promotion must be visible rather than rendered as a clean
    champion.
    """
    out = []
    for r in visible():
        ho = q.artifacts(r["id"]).get("holdout_report")
        status = r["registry_status"]
        if status == "champion" and ho is None:
            out.append({"experiment_id": r["id"], "kind": "CHAMPION_WITHOUT_HOLDOUT",
                        "detail": f"{r['id']} is champion in the registry with no "
                                  "holdout_report.json on disk"})
        if ho is not None and ho.get("promoted") and status not in ("champion", None):
            out.append({"experiment_id": r["id"], "kind": "PROMOTION_NOT_COMMITTED",
                        "detail": f"{r['id']} holdout promoted but registry still "
                                  f"'{status}' — adjudication did not finish"})
    return out


def system() -> dict:
    hb = q.heartbeat() or {}
    live = _in_flight()
    cur = current_cycle()
    db = q.db_status()
    tasks = {}
    try:
        tasks = q.task_counts()
    except q.DataUnavailable:
        pass
    known = authority_available()
    rows = visible()
    sealed = [r for r in authoritative() if r["sealed"]]
    last = max((r["sealed"] for r in sealed), default=None)
    hb_ts = hb.get("timestamp")
    hb_age = _elapsed(hb_ts, None) if hb_ts else None
    enabled = q.autonomy_enabled()

    if live and any(r["state"] == "RUNNING" for r in live):
        autonomy = "RUNNING"
    elif enabled is False:
        autonomy = "DISABLED"
    elif hb.get("status") == "crashed":
        autonomy = "CRASHED"
    elif enabled is None:
        autonomy = "UNKNOWN"
    else:
        autonomy = "IDLE"

    warnings, errors = [], []
    if hb_age is not None and hb_age > HEARTBEAT_STALE_S:
        warnings.append(f"heartbeat stale: {int(hb_age // 3600)}h since last session")
    if hb.get("grid_remaining") == 0:
        warnings.append("research grid exhausted — hypothesis refresh required")
    for r in live:
        if r["state"] in ("STALE", "ORPHANED"):
            errors.append(f"{r['state'].lower()} unsealed experiment directory: {r['id']}")
    if db["status"] != "OK":
        errors.append(f"database {db['status']}: {db.get('detail', '')}")
    for r in rows:
        if r["holdout"] == FAIL:
            art = q.artifacts(r["id"])
            reason = str((art.get("holdout_report") or {}).get("reason") or "")
            if "error" in reason:
                errors.append(f"{r['id']} holdout adjudication error: {reason[:80]}")

    inconsistencies = _lifecycle_inconsistencies()
    for bad in inconsistencies:
        errors.append(bad["detail"])
    freshness = data_freshness()
    eligible_without_artifact = [
        r["id"] for r in rows
        if r["registry_status"] == "eligible" and r["validation"] == NOT_RUN
    ]
    if eligible_without_artifact:
        warnings.append(
            f"{len(eligible_without_artifact)} model(s) marked eligible in the registry "
            "with no validation.json artifact on disk"
        )

    return {
        "autonomy": autonomy,
        "autonomy_enabled": enabled,
        "market": sorted({r["market"] for r in rows if r["market"]}),
        "mode": hb.get("mode") or "OBSERVATION",
        "cycle": cur.get("cycle_number"),
        "cycle_status": cur.get("status"),
        "experiment": cur.get("active_experiment") or cur.get("last_experiment"),
        "stage": cur.get("active_stage"),
        "progress": {"completed": cur.get("completed_experiments"),
                     "planned": cur.get("planned_experiments")},
        # an unsealed directory proves a process started, not that one is alive
        "active_runs": {
            "running": sum(1 for r in live if r["state"] == "RUNNING"),
            "stale": sum(1 for r in live if r["state"] == "STALE"),
            "orphaned": sum(1 for r in live if r["state"] == "ORPHANED"),
            "detail": live,
        },
        "queue": {
            "grid_remaining": hb.get("grid_remaining"),
            "tasks_pending": tasks.get("pending", 0),
            "tasks_failed": tasks.get("failed", 0),
        },
        "last_heartbeat": hb_ts,
        "last_heartbeat_age_s": hb_age,
        "heartbeat_status": hb.get("status") or ("ok" if hb.get("ok") else None),
        "last_experiment_completed": last,
        "experiments_total": len(sealed) if known else NA,
        "population": population(),
        "lifecycle_inconsistencies": inconsistencies,
        "uptime_s": NA,          # no process start time is persisted anywhere
        "data": freshness,
        "database": db,
        "repository": q.git_commit(),
        "scheduled_jobs": q.scheduled_jobs(),
        "errors": errors,
        "warnings": warnings,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


def data_freshness() -> dict:
    p = q.paths()["processed"]
    out = {"status": "UNAVAILABLE", "datasets": []}
    if not p.exists():
        return out
    for f in sorted(p.glob("*.parquet")):
        try:
            rows = q.parquet_rows(f, columns="max(event_time) AS last, count(*) AS n")
        except q.DataUnavailable:
            continue
        last = rows[0]["last"]
        age_days = (datetime.now(timezone.utc).date() - last).days if last else None
        out["datasets"].append({
            "ticker": f.stem, "last_event_time": str(last),
            "rows": rows[0]["n"], "age_days": age_days,
            "file_mtime": q._mtime(f),
        })
    if out["datasets"]:
        worst = max(d["age_days"] or 0 for d in out["datasets"])
        out["status"] = "OK" if worst <= 5 else "STALE"
        out["max_age_days"] = worst
    return out


def activity(limit: int = 120) -> list[dict]:
    """Timestamped events reconstructed from artifacts and session summaries."""
    events: list[dict] = []
    for c in cycles():
        events.append({"t": c["started_at"], "kind": "CYCLE_STARTED", "level": "info",
                       "text": f"Cycle {c['cycle_number']} started · budget {c['planned_experiments']}"})
        if c["completed_at"]:
            events.append({
                "t": c["completed_at"], "kind": "CYCLE_COMPLETED", "level": "info",
                "text": (f"Cycle {c['cycle_number']} completed · {c['completed_experiments']} "
                         f"experiments · {c['keeps']} KEEP / {c['rejects']} REJECT"),
            })
    for r in visible():
        if r["started"]:
            events.append({"t": r["started"], "kind": "EXPERIMENT_STARTED", "level": "info",
                           "text": f"{r['id']} started", "experiment_id": r["id"]})
        if r["sealed"]:
            level = {"KEEP": "pass", "IMPROVE": "warn", "REJECT": "fail"}.get(r["status"], "info")
            events.append({
                "t": r["sealed"], "kind": "EXPERIMENT_COMPLETED", "level": level,
                "text": f"{r['id']} → {r['status']}", "experiment_id": r["id"],
            })
        art = q.artifacts(r["id"])
        if art.get("validated_at"):
            ok = r["validation"] == "APPROVED"
            events.append({
                "t": art["validated_at"],
                "kind": "VALIDATION_PASSED" if ok else "VALIDATION_FAILED",
                "level": "pass" if ok else "fail",
                "text": (f"{r['id']} validation {r['validation']}"
                         + ("" if ok else f" · {len((art.get('validation') or {}).get('issues_found') or [])} issues")),
                "experiment_id": r["id"],
            })
        if art.get("holdout_at"):
            ho = art.get("holdout_report") or {}
            ok = bool(ho.get("promoted"))
            events.append({
                "t": art["holdout_at"],
                "kind": "CHAMPION_PROMOTED" if ok else "HOLDOUT_FAILED",
                "level": "pass" if ok else "fail",
                "text": f"{r['id']} holdout {'PROMOTED' if ok else 'FAILED'} · "
                        f"{holdout_summary(ho)}"[:200],
                "experiment_id": r["id"],
            })
    hb = q.heartbeat()
    if hb and hb.get("timestamp"):
        events.append({"t": hb["timestamp"], "kind": "HEARTBEAT", "level": "info",
                       "text": (f"heartbeat · executed {hb.get('executed')} · "
                                f"grid remaining {hb.get('grid_remaining')}")})
    events = [e for e in events if e["t"]]
    events.sort(key=lambda e: e["t"], reverse=True)
    return events[:limit]


def overview() -> dict:
    all_rows = experiment_index()
    rows = authoritative(all_rows)
    known = authority_available()
    cur = current_cycle()
    done = cycles()
    live = _in_flight()
    hb = q.heartbeat() or {}
    registry = _registry_map(authoritative_only=True)
    evidence = (lambda n: n if known else NA)   # never a filesystem count
    sealed = [r for r in rows if r["status"] not in ("RUNNING", "UNSEALED")]
    hyps = hypotheses()
    default = default_experiment(rows)
    return {
        "market": sorted({r["market"] for r in rows if r["market"]}),
        "cycle": cur,
        "cycles_completed": len(done),
        "progress": {
            # operational: what the machine is doing right now
            "cycles_completed": len(done),
            "experiments_active": sum(1 for r in live if r["state"] == "RUNNING"),
            "experiments_unsealed": len(live),
            "experiments_queued": hb.get("grid_remaining"),
            # evidence: what the research system can stand behind
            "experiments_completed": evidence(len(sealed)),
            "hypotheses_researched": evidence(len(hyps)),
            "hypotheses_rejected": evidence(sum(1 for h in hyps if h["status"] == "REJECTED")),
            "candidates": evidence(sum(1 for m, r in registry.items()
                                       if m.endswith("_improved") and r["status"] == "candidate")),
            "eligible": evidence(sum(1 for m, r in registry.items()
                                     if m.endswith("_improved") and r["status"] == "eligible")),
            "champions": evidence(sum(1 for r in registry.values()
                                      if r["status"] == "champion")),
        },
        "funnel": funnel(),
        "rejections": rejections(),
        "system": system(),
        "default_experiment": default,
        "experiments_total": evidence(len(rows)),
        "runs_visible": len(visible(all_rows)),
        "population": population(),
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
    }


def default_experiment(rows: list[dict] | None = None) -> str | None:
    """The experiment the terminal opens on: champion, else newest sealed."""
    rows = visible(rows if rows is not None else experiment_index())
    for r in rows:
        if r["registry_status"] == "champion":
            return r["id"]
    for r in rows:
        if r["sealed"]:
            return r["id"]
    return rows[0]["id"] if rows else None
