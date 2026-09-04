from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "src" / "quant_loop_trader" / "dashboard" / "service.py"
text = path.read_text()
tree = ast.parse(text)
node = next(
    n for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "population"
)
start = min([d.lineno for d in node.decorator_list], default=node.lineno) - 1
end = node.end_lineno
lines = text.splitlines(keepends=True)
replacement = '''def population() -> dict:
    """Active-market population while preserving the dashboard API contract."""
    all_rows = experiment_index()
    btc = [r for r in all_rows if str(r.get("market") or "").upper() == "BTCUSD"]
    rows = btc or all_rows
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
'''
path.write_text("".join(lines[:start]) + replacement + "".join(lines[end:]))
print("Final dashboard compatibility adjustment applied")
