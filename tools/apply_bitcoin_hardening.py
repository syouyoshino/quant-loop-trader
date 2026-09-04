from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "quant_loop_trader"


def read(path: str | Path) -> str:
    return (ROOT / path).read_text()


def write(path: str | Path, text: str) -> None:
    p = ROOT / path
    p.write_text(text)


def replace_once(path: str | Path, old: str, new: str) -> None:
    text = read(path)
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{path}: expected exactly one occurrence of {old!r}, found {n}")
    write(path, text.replace(old, new, 1))


def replace_all(path: str | Path, old: str, new: str, minimum: int = 1) -> None:
    text = read(path)
    n = text.count(old)
    if n < minimum:
        raise RuntimeError(f"{path}: expected at least {minimum} occurrences of {old!r}, found {n}")
    write(path, text.replace(old, new))


def replace_function(path: str | Path, name: str, source: str) -> None:
    p = ROOT / path
    text = p.read_text()
    tree = ast.parse(text)
    node = next(
        (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name),
        None,
    )
    if node is None or node.end_lineno is None:
        raise RuntimeError(f"{path}: function {name} not found")
    start = min([d.lineno for d in node.decorator_list], default=node.lineno) - 1
    end = node.end_lineno
    lines = text.splitlines(keepends=True)
    replacement = source.rstrip() + "\n"
    p.write_text("".join(lines[:start]) + replacement + "".join(lines[end:]))


def active_python_files():
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC)
        if "experimental" in rel.parts or "migrations" in rel.parts:
            continue
        yield p


# ---------------------------------------------------------------------------
# 1) Remove the 06:00 calendar trigger. Keep the plist as a manual supervisor
#    entry point, but make it unscheduled and able to exhaust the real frontier.
# ---------------------------------------------------------------------------
plist_path = "deploy/com.quantloop.research-session.plist"
plist = read(plist_path)
plist, n = re.subn(
    r"\n\s*<key>StartCalendarInterval</key>\s*\n\s*<dict>\s*\n"
    r"\s*<key>Hour</key>\s*\n\s*<integer>6</integer>\s*\n"
    r"\s*<key>Minute</key>\s*\n\s*<integer>0</integer>\s*\n\s*</dict>",
    "",
    plist,
    count=1,
)
if n != 1:
    raise RuntimeError("research-session plist: 06:00 StartCalendarInterval block not found")
# Manual supervisor launch should not inherit the old one-experiment batch limit.
plist, n = re.subn(
    r"(<string>--max-experiments</string>\s*\n\s*<string>)1(</string>)",
    r"\g<1>100\g<2>",
    plist,
    count=1,
)
if n != 1:
    raise RuntimeError("research-session plist: max-experiments=1 not found")
write(plist_path, plist)


# ---------------------------------------------------------------------------
# 2) Bitcoin-first active runtime. Historical SPY logic/data remains intact;
#    only dangerous/default ticker assumptions are changed.
# ---------------------------------------------------------------------------
for p in active_python_files():
    text = p.read_text()
    text = text.replace('ticker: str = "SPY"', 'ticker: str = "BTCUSD"')
    text = text.replace("ticker: str = 'SPY'", "ticker: str = 'BTCUSD'")
    text = text.replace('cfg.get("ticker", "SPY")', 'cfg.get("ticker", "BTCUSD")')
    text = text.replace("cfg.get('ticker', 'SPY')", "cfg.get('ticker', 'BTCUSD')")
    text = text.replace('cfg.get("ticker") or "SPY"', 'cfg.get("ticker") or "BTCUSD"')
    text = text.replace("cfg.get('ticker') or 'SPY'", "cfg.get('ticker') or 'BTCUSD'")
    text = text.replace('p.add_argument("--ticker", default="SPY")', 'p.add_argument("--ticker", default="BTCUSD")')
    p.write_text(text)

# Mandatory market identity at scientific entry points where compatibility cost is low.
replace_once(
    "src/quant_loop_trader/autonomy.py",
    'def _frontier_remaining(ticker: str = "BTCUSD", horizon: int = 5) -> int:',
    'def _frontier_remaining(ticker: str, horizon: int = 5) -> int:',
)
replace_once(
    "src/quant_loop_trader/autonomy.py",
    'def review_memory(ticker: str = "BTCUSD", horizon: int = 5) -> dict:',
    'def review_memory(ticker: str, horizon: int = 5) -> dict:',
)
replace_once(
    "src/quant_loop_trader/autonomy.py",
    'def run_session(ticker: str = "BTCUSD", horizon: int = 5,',
    'def run_session(ticker: str, horizon: int = 5,',
)
replace_once(
    "src/quant_loop_trader/experiment.py",
    'horizon: int, seed: int, ticker: str = "BTCUSD") -> dict:',
    'horizon: int, seed: int, ticker: str) -> dict:',
)
replace_once(
    "src/quant_loop_trader/experiment.py",
    'def run_experiment(ticker: str = "BTCUSD", horizon: int = 5,',
    'def run_experiment(ticker: str, horizon: int = 5,',
)
replace_once(
    "src/quant_loop_trader/data.py",
    'def fetch_ohlcv(ticker: str = "BTCUSD", start: str = "2018-01-01", end: str = "2024-12-31",',
    'def fetch_ohlcv(ticker: str, start: str = "2018-01-01", end: str = "2024-12-31",',
)

# CLI/dashboard launches should naturally run the whole unexplored frontier.
replace_once(
    "src/quant_loop_trader/autonomy.py",
    'max_experiments: int = 3, validate: bool = True) -> dict:',
    'max_experiments: int = 100, validate: bool = True) -> dict:',
)
replace_once(
    "src/quant_loop_trader/autonomy.py",
    'p.add_argument("--max-experiments", type=int, default=3)',
    'p.add_argument("--max-experiments", type=int, default=100)',
)
replace_once(
    "src/quant_loop_trader/dashboard/api.py",
    'max_experiments = int(payload.get("max_experiments", 3))',
    'max_experiments = int(payload.get("max_experiments", 100))',
)
replace_once(
    "dashboard/src/components/control.js",
    'name="max_experiments" type="number" min="1" max="100" value="3"',
    'name="max_experiments" type="number" min="1" max="100" value="100"',
)


# ---------------------------------------------------------------------------
# 3) Weekly report: ticker + campaign scoped, including the candidate frontier.
# ---------------------------------------------------------------------------
replace_once(
    "src/quant_loop_trader/report.py",
    "from quant_loop_trader.autonomy import GRID, _already_run",
    "from quant_loop_trader.autonomy import _already_run, _candidate_grid",
)
replace_once(
    "src/quant_loop_trader/report.py",
    "from quant_loop_trader.experiment import EXP_ROOT",
    "from quant_loop_trader.experiment import EXP_ROOT\nfrom quant_loop_trader.market import campaign_id",
)
replace_function(
    "src/quant_loop_trader/report.py",
    "_frontier",
    '''def _frontier(ticker: str, horizon: int = 5) -> dict:
    grid = _candidate_grid(ticker)
    total = len(grid)
    done = sum(
        1
        for c in grid
        if _already_run(ticker, horizon, c["start"], c["end"], c["seed"])
    )
    return {"grid_total": total, "explored": done, "remaining": total - done}''',
)
replace_function(
    "src/quant_loop_trader/report.py",
    "generate_report",
    '''def generate_report(
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
    beliefs = con.execute(
        "SELECT hypothesis, memory_type, confidence, created_at FROM research_memory m "
        "WHERE authoritative AND upper(coalesce(ticker, ''))=? "
        "AND created_at = (SELECT max(created_at) FROM research_memory x "
        "WHERE x.hypothesis=m.hypothesis AND x.authoritative "
        "AND upper(coalesce(x.ticker, ''))=?) "
        "ORDER BY confidence DESC LIMIT 10",
        [ticker, ticker],
    ).fetchall()
    lessons = con.execute(
        "SELECT lesson, outcome, created_at FROM research_memory "
        "WHERE authoritative AND upper(coalesce(ticker, ''))=? "
        "AND memory_type IN ('failure','success','partial') "
        "ORDER BY created_at DESC LIMIT 8",
        [ticker],
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
    out = out_dir / f"weekly_{ticker}_{now.strftime('%G-W%V')}.md"
    out.write_text("\\n".join(lines) + "\\n")
    logger.info(json.dumps({"event": "report_written", "path": str(out), "ticker": ticker}))
    return out''',
)


# ---------------------------------------------------------------------------
# 4) Dashboard is globally BTCUSD-scoped at the read layer. SPY remains in
#    DuckDB/filesystem but is excluded from programme-level dashboard views.
# ---------------------------------------------------------------------------
queries_path = "src/quant_loop_trader/dashboard/queries.py"
qtext = read(queries_path)
anchor = 'CACHE_TTL = float(os.environ.get("QLT_DASHBOARD_CACHE_TTL", "5"))\n'
if 'DASHBOARD_MARKET = ' not in qtext:
    if anchor not in qtext:
        raise RuntimeError("dashboard queries: CACHE_TTL anchor not found")
    qtext = qtext.replace(
        anchor,
        anchor + 'DASHBOARD_MARKET = (os.environ.get("QLT_DASHBOARD_MARKET", "BTCUSD").strip().upper() or "BTCUSD")\n',
        1,
    )
    write(queries_path, qtext)

replace_function(
    queries_path,
    "experiment_rows",
    '''@ttl_cache
def experiment_rows() -> list[dict]:
    return query(
        "SELECT experiment_id, dataset_id, ticker, horizon_days, hypothesis, "
        "economic_reasoning, research_question, model_version, feature_version, seed, "
        "config_json, metrics_json, decision, parent_experiment_id, created_at, "
        "authoritative FROM experiments WHERE upper(ticker)=? ORDER BY created_at",
        [DASHBOARD_MARKET],
    )''',
)
replace_function(
    queries_path,
    "authoritative_ids",
    '''@ttl_cache
def authoritative_ids() -> tuple[set[str], set[str]] | None:
    """BTC-scoped (authoritative stems, quarantined stems)."""
    try:
        rows = query(
            "SELECT experiment_id, authoritative FROM experiments WHERE upper(ticker)=?",
            [DASHBOARD_MARKET],
        )
    except DataUnavailable:
        return None
    good = {stem(r["experiment_id"]) for r in rows if r["authoritative"]}
    return good, {stem(r["experiment_id"]) for r in rows} - good''',
)
replace_function(
    queries_path,
    "model_registry_rows",
    '''@ttl_cache
def model_registry_rows() -> list[dict]:
    rows = query(
        "SELECT model_id, parent_model_id, training_data_version, feature_version, "
        "status, research_lineage, performance_history_json, failure_modes, created_at "
        "FROM model_registry ORDER BY created_at"
    )
    allowed = {stem(r["experiment_id"]) for r in experiment_rows()}
    return [r for r in rows if stem(r["model_id"]) in allowed]''',
)
replace_function(
    queries_path,
    "research_memory_rows",
    '''@ttl_cache
def research_memory_rows() -> list[dict]:
    return query(
        "SELECT memory_id, experiment_id, memory_type, hypothesis, outcome, lesson, "
        "confidence, created_at, authoritative FROM research_memory "
        "WHERE authoritative AND upper(coalesce(ticker, ''))=? ORDER BY created_at DESC",
        [DASHBOARD_MARKET],
    )''',
)
replace_function(
    queries_path,
    "dataset_rows",
    '''@ttl_cache
def dataset_rows() -> list[dict]:
    return query(
        "SELECT dataset_id, ticker, start_date, end_date, source, row_count, "
        "validation_status, checksum, created_at FROM datasets "
        "WHERE upper(ticker)=? ORDER BY created_at",
        [DASHBOARD_MARKET],
    )''',
)
replace_function(
    queries_path,
    "experiment_ids",
    '''@ttl_cache
def experiment_ids() -> list[str]:
    exp_root = paths()["experiments"]
    if not exp_root.exists():
        return []
    marker = f"_{DASHBOARD_MARKET}_"
    return sorted(p.name for p in exp_root.iterdir() if p.is_dir() and marker in p.name)''',
)
replace_function(
    queries_path,
    "price_history",
    '''@ttl_cache
def price_history(ticker: str = "BTCUSD") -> list[dict]:
    ticker = (ticker or DASHBOARD_MARKET).upper()
    return parquet_rows(paths()["processed"] / f"{ticker}.parquet", order="event_time")''',
)
replace_function(
    queries_path,
    "session_records",
    '''@ttl_cache
def session_records() -> list[dict]:
    """BTC-scoped autonomy session summaries from session.log."""
    log = paths()["logs"] / "session.log"
    if not log.exists():
        return []
    text = log.read_text()
    decoder = json.JSONDecoder()
    out, i = [], 0
    while i < len(text):
        start = text.find("{", i)
        if start < 0:
            break
        try:
            obj, i = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        ticker = str((obj.get("memory_review") or {}).get("ticker") or "").upper() if isinstance(obj, dict) else ""
        if isinstance(obj, dict) and "session_started" in obj and ticker == DASHBOARD_MARKET:
            out.append(obj)
    out.sort(key=lambda r: r["session_started"])
    return out''',
)

# Make the UI send the BTC filter from first paint, not only after a run starts.
replace_once(
    "dashboard/src/pages/terminal.js",
    "  filters: {},\n  perf: null,",
    "  filters: { market: 'BTCUSD' },\n  perf: null,",
)

# The backend service has a couple of compatibility fallbacks; make them BTC-first.
service_path = "src/quant_loop_trader/dashboard/service.py"
service = read(service_path)
service = service.replace('ticker: str = "SPY"', 'ticker: str = "BTCUSD"')
service = service.replace('cfg.get("ticker") or "SPY"', 'cfg.get("ticker") or "BTCUSD"')
write(service_path, service)

# Dashboard docs should describe the actual default market snapshot.
dashboard_doc = read("docs/dashboard.md").replace(
    "`data/processed/SPY.parquet` (research snapshot, not a live feed)",
    "`data/processed/BTCUSD.parquet` (research snapshot, not a live feed)",
)
write("docs/dashboard.md", dashboard_doc)

# Explain the launcher model next to the controls documentation.
controls_path = "docs/dashboard_controls.md"
controls = read(controls_path)
launcher_note = '''\n## Launch model\n\nThe research-session launchd plist has no calendar trigger. Research starts only from the\nlocalhost dashboard controls or an explicit supervisor/manual launch. The default session\nbudget is 100, which is deliberately larger than the normal candidate frontier; the\nanti-duplicate `_already_run` check and campaign identity remain authoritative, so a\nsession naturally stops when there is no genuinely unexplored evidence left.\n'''
if "## Launch model" not in controls:
    controls += launcher_note
write(controls_path, controls)


# ---------------------------------------------------------------------------
# 5) CI guardrail: active runtime may retain SPY support, but no function may
#    silently default a ticker parameter to SPY again.
# ---------------------------------------------------------------------------
ci_test = r'''from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "quant_loop_trader"


def _active_python_files():
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        if "experimental" in rel.parts or "migrations" in rel.parts:
            continue
        yield path


def _spy_ticker_defaults(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    failures = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        positional = list(node.args.posonlyargs) + list(node.args.args)
        defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
        for arg, default in zip(positional, defaults):
            if arg.arg == "ticker" and isinstance(default, ast.Constant) and default.value == "SPY":
                failures.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            if arg.arg == "ticker" and isinstance(default, ast.Constant) and default.value == "SPY":
                failures.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
    return failures


def test_active_runtime_never_defaults_ticker_to_spy():
    failures = [item for path in _active_python_files() for item in _spy_ticker_defaults(path)]
    assert failures == [], "active runtime SPY ticker defaults:\n" + "\n".join(failures)


def test_research_plist_has_no_calendar_trigger():
    text = (ROOT / "deploy" / "com.quantloop.research-session.plist").read_text()
    assert "StartCalendarInterval" not in text
    assert "--ticker" in text and "BTCUSD" in text


def test_scientific_entrypoints_require_ticker():
    from quant_loop_trader import autonomy, data, experiment

    required = [
        autonomy.run_session,
        autonomy._frontier_remaining,
        autonomy.review_memory,
        data.fetch_ohlcv,
        experiment.run_experiment,
        experiment.train_evaluate_from,
    ]
    for fn in required:
        sig = __import__("inspect").signature(fn)
        assert sig.parameters["ticker"].default is __import__("inspect").Parameter.empty


def test_dashboard_defaults_to_btcusd():
    from quant_loop_trader.dashboard import queries

    assert queries.DASHBOARD_MARKET == "BTCUSD"
    assert queries.price_history.__wrapped__.__defaults__ == ("BTCUSD",) if hasattr(queries.price_history, "__wrapped__") else True
    js = (ROOT / "dashboard" / "src" / "pages" / "terminal.js").read_text()
    assert "filters: { market: 'BTCUSD' }" in js
'''
write("tests/test_bitcoin_defaults.py", ci_test)

# Existing dashboard-control test snapshots the prior default batch size.
tdc = read("tests/test_dashboard_controls.py")
tdc = tdc.replace('"max_experiments": 3,', '"max_experiments": 100,')
write("tests/test_dashboard_controls.py", tdc)

# Final source-level invariant before CI even starts.
violations = []
for p in active_python_files():
    tree = ast.parse(p.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        positional = list(node.args.posonlyargs) + list(node.args.args)
        defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
        for arg, default in zip(positional, defaults):
            if arg.arg == "ticker" and isinstance(default, ast.Constant) and default.value == "SPY":
                violations.append(f"{p.relative_to(ROOT)}:{node.lineno}:{node.name}")
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            if arg.arg == "ticker" and isinstance(default, ast.Constant) and default.value == "SPY":
                violations.append(f"{p.relative_to(ROOT)}:{node.lineno}:{node.name}")
if violations:
    raise RuntimeError("SPY ticker defaults remain:\n" + "\n".join(violations))

print("Bitcoin hardening transformations applied")
