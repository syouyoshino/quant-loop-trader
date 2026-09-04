from __future__ import annotations

import ast
from pathlib import Path


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
    from quant_loop_trader import data, experiment

    required = [
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


def test_orchestration_defaults_are_btcusd():
    import inspect
    from quant_loop_trader import autonomy

    for fn in (autonomy.run_session, autonomy._frontier_remaining, autonomy.review_memory):
        assert inspect.signature(fn).parameters["ticker"].default == "BTCUSD"
