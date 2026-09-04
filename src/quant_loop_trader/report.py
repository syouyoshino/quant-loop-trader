"""Weekly research report — reads DuckDB + memory, writes markdown. No new architecture."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from quant_loop_trader import data as _data
from quant_loop_trader.data import migrate_db
from quant_loop_trader.experiment import EXP_ROOT
from quant_loop_trader.market import campaign_id
from quant_loop_trader.autonomy import _already_run, _candidate_grid

logger = logging.getLogger(__name__)


def _frontier(ticker: str, horizon: int = 5) -> dict:
    grid = _candidate_grid(ticker)
    total = len(grid)
    done = sum(
        1
        for c in grid
        if _already_run(ticker, horizon, c["start"], c["end"], c["seed"])
    )
    return {"grid_total": total, "explored": done, "remaining": total - done}


def generate_report(
    out_dir: Path | None = None,
    ticker: str = "BTCUSD",
    horizon: int = 5,
) -> Path:
    ticker = ticker.upper()
    migrate_db()
    con = duckdb.connect(str(_data.DB_PATH))

    exp_total = con.execute(
        "SELECT count(*) FROM experiments WHERE authoritative AND ticker=?",
        [ticker],
    ).fetchone()[0]
    decisions = con.execute(
        "SELECT decision, count(*) FROM experiments "
        "WHERE authoritative AND ticker=? AND experiment_id NOT LIKE '%_baseline' "
        "GROUP BY decision",
        [ticker],
    ).fetchall()
    week_exps = con.execute(
        "SELECT experiment_id, decision FROM experiments "
        "WHERE authoritative AND ticker=? "
        "AND created_at >= current_timestamp - INTERVAL 7 DAY "
        "AND experiment_id NOT LIKE '%_baseline' ORDER BY created_at DESC",
        [ticker],
    ).fetchall()
    memory_pattern = f"%_{ticker}_%"
    beliefs = con.execute(
        "SELECT hypothesis, memory_type, confidence, created_at FROM research_memory m "
        "WHERE authoritative AND experiment_id LIKE ? "
        "AND created_at = (SELECT max(created_at) FROM research_memory x "
        "WHERE x.hypothesis=m.hypothesis AND x.authoritative AND x.experiment_id LIKE ?) "
        "ORDER BY confidence DESC LIMIT 10",
        [memory_pattern, memory_pattern],
    ).fetchall()
    lessons = con.execute(
        "SELECT lesson, outcome, created_at FROM research_memory "
        "WHERE authoritative AND experiment_id LIKE ? "
        "AND memory_type IN ('failure','success','partial') "
        "ORDER BY created_at DESC LIMIT 8",
        [memory_pattern],
    ).fetchall()
    datasets = con.execute(
        "SELECT dataset_id, ticker, row_count, validation_status, checksum "
        "FROM datasets WHERE upper(ticker)=?",
        [ticker],
    ).fetchall()
    models = con.execute(
        "SELECT status, count(*) FROM model_registry WHERE model_id LIKE ? GROUP BY status",
        [f"%_{ticker}_%"],
    ).fetchall()
    con.close()

    frontier = _frontier(ticker, horizon)
    now = datetime.now(timezone.utc)
    cid = campaign_id(ticker)
    lines = [
        f"# Weekly Research Report — {now.strftime('%G-W%V')}",
        f"_Generated: {now.isoformat()} UTC. Market: {ticker}. Campaign: {cid}. "
        "Mode: OBSERVATION. No champion promotion without human approval._",
        "",
        "## Experiment activity",
        f"- Total experiment variants stored: {exp_total}",
        "- Decision breakdown: " + ", ".join(f"{d}: {n}" for d, n in decisions),
        f"- Last 7 days: {len(week_exps)} completed hypotheses"
        + ("" if week_exps else " (idle — frontier exhausted or stopped)"),
    ]
    for eid, dec in week_exps[:10]:
        vfile = EXP_ROOT / eid.replace("_improved", "") / "validation.json"
        vstatus = "n/a"
        if vfile.exists():
            vstatus = json.loads(vfile.read_text()).get("approval_status", "n/a")
        lines.append(f"  - `{eid}` → {dec} (validation: {vstatus})")

    lines += ["", "## Belief state (latest confidence per hypothesis)"]
    for hyp, mtype, conf, _ in beliefs:
        lines.append(f"- [{mtype}] {conf:.2f} — {hyp[:80]}")

    lines += ["", "## Recent lessons"]
    for lesson, outcome, _ in lessons:
        lines.append(f"- ({outcome}) {lesson[:140]}")

    lines += ["", "## Model registry"]
    lines.append("- " + ", ".join(f"{s}: {n}" for s, n in models))

    lines += ["", "## Data health"]
    for ds in datasets:
        lines.append(f"- {ds[0]} rows={ds[2]} status={ds[3]} checksum={ds[4]}")

    lines += ["", f"## Research frontier — {ticker} / {cid}"]
    lines.append(
        f"- Grid explored: {frontier['explored']}/{frontier['grid_total']} "
        f"(remaining {frontier['remaining']})"
    )
    lines.append("- When remaining=0 the loop idles by design: no duplicate mining.")

    out_dir = Path(out_dir or Path.cwd() / "data" / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"weekly_{now.strftime('%G-W%V')}.md"
    out.write_text("\n".join(lines) + "\n")
    logger.info(json.dumps({"event": "report_written", "path": str(out), "ticker": ticker}))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(generate_report())
