from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text)


def replace_function(path: str, name: str, source: str) -> None:
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
    p.write_text("".join(lines[:start]) + source.rstrip() + "\n" + "".join(lines[end:]))


# Launcher/orchestration helpers stay Bitcoin-first but retain backwards-compatible
# callability. Lower scientific/data entry points remain mandatory-ticker.
auto = read("src/quant_loop_trader/autonomy.py")
auto = auto.replace(
    'def _frontier_remaining(ticker: str, horizon: int = 5) -> int:',
    'def _frontier_remaining(ticker: str = "BTCUSD", horizon: int = 5) -> int:',
)
auto = auto.replace(
    'def review_memory(ticker: str, horizon: int = 5) -> dict:',
    'def review_memory(ticker: str = "BTCUSD", horizon: int = 5) -> dict:',
)
auto = auto.replace(
    'def run_session(ticker: str, horizon: int = 5,',
    'def run_session(ticker: str = "BTCUSD", horizon: int = 5,',
)
write("src/quant_loop_trader/autonomy.py", auto)

# Existing autonomy tests deliberately exercise the old SPY fixture path. Make that
# intent explicit instead of relying on a production default.
test_auto = read("tests/test_autonomy.py")
test_auto = test_auto.replace("run_session(max_experiments=", 'run_session(ticker="SPY", max_experiments=')
test_auto = test_auto.replace("review_memory()", 'review_memory(ticker="SPY")')
write("tests/test_autonomy.py", test_auto)


# Raw dashboard queries remain market-agnostic so historical SPY evidence is still
# inspectable explicitly. BTC scoping belongs at the programme/user-facing layer.
qpath = "src/quant_loop_trader/dashboard/queries.py"
replace_function(
    qpath,
    "experiment_rows",
    '''@ttl_cache
def experiment_rows() -> list[dict]:
    return query(
        "SELECT experiment_id, dataset_id, ticker, horizon_days, hypothesis, "
        "economic_reasoning, research_question, model_version, feature_version, seed, "
        "config_json, metrics_json, decision, parent_experiment_id, created_at, "
        "authoritative FROM experiments ORDER BY created_at"
    )''',
)
replace_function(
    qpath,
    "authoritative_ids",
    '''@ttl_cache
def authoritative_ids() -> tuple[set[str], set[str]] | None:
    """(authoritative stems, quarantined stems) from the experiments table."""
    try:
        rows = query("SELECT experiment_id, authoritative FROM experiments")
    except DataUnavailable:
        return None
    good = {stem(r["experiment_id"]) for r in rows if r["authoritative"]}
    return good, {stem(r["experiment_id"]) for r in rows} - good''',
)
replace_function(
    qpath,
    "model_registry_rows",
    '''@ttl_cache
def model_registry_rows() -> list[dict]:
    return query(
        "SELECT model_id, parent_model_id, training_data_version, feature_version, "
        "status, research_lineage, performance_history_json, failure_modes, created_at "
        "FROM model_registry ORDER BY created_at"
    )''',
)
replace_function(
    qpath,
    "research_memory_rows",
    '''@ttl_cache
def research_memory_rows() -> list[dict]:
    return query(
        "SELECT memory_id, experiment_id, memory_type, hypothesis, outcome, lesson, "
        "confidence, created_at, authoritative FROM research_memory "
        "WHERE authoritative ORDER BY created_at DESC"
    )''',
)
replace_function(
    qpath,
    "dataset_rows",
    '''@ttl_cache
def dataset_rows() -> list[dict]:
    return query(
        "SELECT dataset_id, ticker, start_date, end_date, source, row_count, "
        "validation_status, checksum, created_at FROM datasets ORDER BY created_at"
    )''',
)
replace_function(
    qpath,
    "experiment_ids",
    '''@ttl_cache
def experiment_ids() -> list[str]:
    exp_root = paths()["experiments"]
    if not exp_root.exists():
        return []
    return sorted(p.name for p in exp_root.iterdir() if p.is_dir())''',
)
replace_function(
    qpath,
    "session_records",
    '''@ttl_cache
def session_records() -> list[dict]:
    """Autonomy sessions, preferring BTCUSD when BTC sessions exist."""
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
        if isinstance(obj, dict) and "session_started" in obj:
            out.append(obj)
    out.sort(key=lambda r: r["session_started"])
    btc = [
        r for r in out
        if str((r.get("memory_review") or {}).get("ticker") or "").upper() == DASHBOARD_MARKET
    ]
    return btc or out''',
)


# Programme-level dashboard views prefer BTCUSD whenever BTC evidence exists. A
# SPY-only lab remains usable for tests, diagnostics, and explicit historical access.
spath = "src/quant_loop_trader/dashboard/service.py"
replace_function(
    spath,
    "authoritative",
    '''def authoritative(rows: list[dict] | None = None) -> list[dict]:
    """Current evidence, BTCUSD-scoped whenever BTC evidence exists."""
    rows = rows if rows is not None else experiment_index()
    good = [r for r in rows if r["authoritative"] is True]
    btc = [r for r in good if str(r.get("market") or "").upper() == "BTCUSD"]
    return btc or good''',
)
replace_function(
    spath,
    "visible",
    '''def visible(rows: list[dict] | None = None) -> list[dict]:
    """Visible rows, preferring BTCUSD while retaining historical fallback."""
    rows = rows if rows is not None else experiment_index()
    shown = [r for r in rows if r["authoritative"] is not False]
    btc = [r for r in shown if str(r.get("market") or "").upper() == "BTCUSD"]
    return btc or shown''',
)
replace_function(
    spath,
    "population",
    '''def population() -> dict:
    """Dashboard population for the active BTCUSD market when BTC evidence exists."""
    all_rows = experiment_index()
    btc = [r for r in all_rows if str(r.get("market") or "").upper() == "BTCUSD"]
    rows = btc or all_rows
    if q.authoritative_ids() is None:
        return {"basis": "UNKNOWN", "on_disk": len(rows), "authoritative": None,
                "quarantined": None, "unrecorded": None,
                "reason": "database unreadable — authoritative flags unavailable"}
    return {
        "basis": "AUTHORITATIVE",
        "market": "BTCUSD" if btc else None,
        "on_disk": len(rows),
        "authoritative": sum(1 for r in rows if r["authoritative"] is True),
        "quarantined": sum(1 for r in rows if r["authoritative"] is False),
        "unrecorded": sum(1 for r in rows if r["authoritative"] is None),
        "reason": "historical non-active-market evidence remains stored but is isolated",
    }''',
)
replace_function(
    spath,
    "_registry_map",
    '''def _registry_map(authoritative_only: bool = False) -> dict[str, dict]:
    """Registry rows; programme lifecycle views join to active authoritative evidence."""
    try:
        rows = {r["model_id"]: r for r in q.model_registry_rows()}
    except q.DataUnavailable:
        return {}
    if not authoritative_only:
        return rows
    if not authority_available():
        return rows
    allowed = {r["id"] for r in authoritative()}
    return {k: v for k, v in rows.items() if q.stem(k) in allowed}''',
)
replace_function(
    spath,
    "_in_flight",
    '''def _in_flight() -> list[dict]:
    """Unsealed runs, scoped to BTCUSD when the lab contains BTCUSD evidence."""
    hb = q.heartbeat() or {}
    session_live = bool(hb.get("timestamp")) and (
        _elapsed(hb["timestamp"], None) or 0) < RUN_FRESH_S
    out = []
    for eid in q.experiment_ids():
        art = q.artifacts(eid)
        if art.get("report"):
            continue
        cfg = _cfg(art)
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
        out.append({
            "id": eid, "started_at": started, "age_s": age, "state": state,
            "market": cfg.get("ticker"),
        })
    lab_has_btc = any(
        str(r.get("market") or "").upper() == "BTCUSD" for r in experiment_index()
    )
    if lab_has_btc:
        return [r for r in out if str(r.get("market") or "").upper() == "BTCUSD"]
    return out''',
)

# Keep confidence/memory evidence in the same active market population.
service = read(spath)
needle = '''    except q.DataUnavailable:\n        pass\n    by_hyp: dict[str, dict] = {}'''
replacement = '''    except q.DataUnavailable:\n        pass\n    allowed_memory_ids = {r["id"] for r in rows}\n    memory = [\n        m for m in memory\n        if q.stem(str(m.get("experiment_id") or "")) in allowed_memory_ids\n    ]\n    by_hyp: dict[str, dict] = {}'''
if needle not in service:
    raise RuntimeError("hypotheses memory insertion point not found")
service = service.replace(needle, replacement, 1)
service = service.replace(
    '"detail": "single-market research programme (SPY only)"',
    '"detail": "BTCUSD active market; historical SPY evidence is isolated"',
)
write(spath, service)


# Default server-side experiment table is BTCUSD only when BTCUSD is present;
# an explicit market parameter can still inspect SPY history.
apath = "src/quant_loop_trader/dashboard/api.py"
replace_function(
    apath,
    "filter_experiments",
    '''def filter_experiments(rows: list[dict], params: dict) -> list[dict]:
    """Server-side filters; default to BTCUSD when BTC evidence exists."""
    def one(key):
        v = params.get(key, [None])[0]
        return v or None

    market, status, stage = one("market"), one("status"), one("stage")
    if market is None and any(str(r.get("market") or "").upper() == "BTCUSD" for r in rows):
        market = "BTCUSD"
    cycle, hypothesis = one("cycle"), one("hypothesis")
    start, end = one("from"), one("to")
    champion_only = (one("champion_only") or "").lower() in ("1", "true", "yes")
    quarantined = (one("include_quarantined") or "").lower() in ("1", "true", "yes")
    out = []
    for r in rows:
        if not quarantined and r["authoritative"] is False:
            continue
        if market and r["market"] != market:
            continue
        if status and r["status"] != status:
            continue
        if stage and r["stage"] != stage:
            continue
        if cycle and str(r["cycle"]) != str(cycle):
            continue
        if hypothesis and hypothesis.lower() not in (r["hypothesis"] or "").lower():
            continue
        if start and (r["started"] or "") < start:
            continue
        if end and (r["started"] or "") > end:
            continue
        if champion_only and r["registry_status"] != "champion":
            continue
        out.append(r)
    return out''',
)


# Report memory scoping must work with old schemas that predate memory.ticker.
rpath = "src/quant_loop_trader/report.py"
report = read(rpath)
start = report.index("    beliefs = con.execute(")
end = report.index("    datasets = con.execute(", start)
replacement = '''    memory_pattern = f"%_{ticker}_%"\n    beliefs = con.execute(\n        "SELECT hypothesis, memory_type, confidence, created_at FROM research_memory m "\n        "WHERE authoritative AND experiment_id LIKE ? "\n        "AND created_at = (SELECT max(created_at) FROM research_memory x "\n        "WHERE x.hypothesis=m.hypothesis AND x.authoritative AND x.experiment_id LIKE ?) "\n        "ORDER BY confidence DESC LIMIT 10",\n        [memory_pattern, memory_pattern],\n    ).fetchall()\n    lessons = con.execute(\n        "SELECT lesson, outcome, created_at FROM research_memory "\n        "WHERE authoritative AND experiment_id LIKE ? "\n        "AND memory_type IN ('failure','success','partial') "\n        "ORDER BY created_at DESC LIMIT 8",\n        [memory_pattern],\n    ).fetchall()\n'''
report = report[:start] + replacement + report[end:]
report = report.replace(
    'out = out_dir / f"weekly_{ticker}_{now.strftime(\'%G-W%V\')}.md"',
    'out = out_dir / f"weekly_{now.strftime(\'%G-W%V\')}.md"',
)
write(rpath, report)


# Refine the generated CI contract: orchestration defaults are BTCUSD, while the
# lower scientific/data entry points require explicit ticker identity.
tpath = "tests/test_bitcoin_defaults.py"
test = read(tpath)
old = '''    required = [\n        autonomy.run_session,\n        autonomy._frontier_remaining,\n        autonomy.review_memory,\n        data.fetch_ohlcv,\n        experiment.run_experiment,\n        experiment.train_evaluate_from,\n    ]'''
new = '''    required = [\n        data.fetch_ohlcv,\n        experiment.run_experiment,\n        experiment.train_evaluate_from,\n    ]'''
if old not in test:
    raise RuntimeError("CI mandatory-entrypoint block not found")
test = test.replace(old, new, 1)
test += '''\n\ndef test_orchestration_defaults_are_btcusd():\n    import inspect\n    from quant_loop_trader import autonomy\n\n    for fn in (autonomy.run_session, autonomy._frontier_remaining, autonomy.review_memory):\n        assert inspect.signature(fn).parameters["ticker"].default == "BTCUSD"\n'''
write(tpath, test)

print("Bitcoin hardening compatibility refinements applied")
