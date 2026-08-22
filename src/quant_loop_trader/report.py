"""Weekly research report — reads DuckDB + memory, writes markdown. No new architecture."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from quant_loop_trader.data import DB_PATH, migrate_db, dataset_metadata
from quant_loop_trader.experiment import EXP_ROOT
from quant_loop_trader.research_memory import search_memory
from quant_loop_trader.autonomy import GRID, _already_run

logger = logging.getLogger(__name__)


def _frontier(ticker: str = "SPY", horizon: int = 5) -> dict:
    total = len(GRID)
    done = sum(1 for c in GRID if _already_run(ticker, horizon, c["start"], c["end"], c["seed"]))
    return {"grid_total": total, "explored": done, "remaining": total - done}


def generate_report(out_dir: Path | None = None) -> Path:
    migrate_db()
    con = duckdb.connect(str(DB_PATH))

    exp_total = con.execute("SELECT count(*) FROM experiments").fetchone()[0]
    decisions = con.execute(
        "SELECT decision, count(*) FROM experiments WHERE experiment_id NOT LIKE '%_baseline' GROUP BY decision"
    ).fetchall()
    week_exps = con.execute(
        "SELECT experiment_id, decision FROM experiments "
        "WHERE created_at >= current_timestamp - INTERVAL 7 DAY AND experiment_id NOT LIKE '%_baseline' "
        "ORDER BY created_at DESC"
    ).fetchall()
    beliefs = con.execute(
        "SELECT hypothesis, memory_type, confidence, created_at FROM research_memory m "
        "WHERE created_at = (SELECT max(created_at) FROM research_memory x WHERE x.hypothesis = m.hypothesis) "
        "ORDER BY confidence DESC LIMIT 10"
    ).fetchall()
    lessons = con.execute(
        "SELECT lesson, outcome, created_at FROM research_memory "
        "WHERE memory_type IN ('failure','success','partial') ORDER BY created_at DESC LIMIT 8"
    ).fetchall()
    datasets = con.execute("SELECT dataset_id, ticker, row_count, validation_status, checksum FROM datasets").fetchall()
    models = con.execute("SELECT status, count(*) FROM model_registry GROUP BY status").fetchall()
    con.close()

    frontier = _frontier()
    now = datetime.now(timezone.utc)
    lines = [
        f"# Weekly Research Report — {now.strftime('%G-W%V')}",
        f"_Generated: {now.isoformat()} UTC. Mode: OBSERVATION. No champion promotion without human approval._",
        "",
        "## Experiment activity",
        f"- Total experiment variants stored: {exp_total}",
        "- Decision breakdown: " + ", ".join(f"{d}: {n}" for d, n in decisions),
        f"- Last 7 days: {len(week_exps)} completed hypotheses" + ("" if week_exps else " (idle — grid exhausted or budget)"),
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

    lines += ["", "## Research frontier (anti-mining governor)"]
    lines.append(f"- Grid explored: {frontier['explored']}/{frontier['grid_total']} (remaining {frontier['remaining']})")
    lines.append("- When remaining=0 the loop idles by design: no duplicate mining.")

    out_dir = Path(out_dir or Path.cwd() / "data" / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"weekly_{now.strftime('%Y-W%V')}.md"
    out.write_text("\n".join(lines) + "\n")
    logger.info(json.dumps({"event": "report_written", "path": str(out)}))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(generate_report())
