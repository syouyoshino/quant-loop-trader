# Dashboard research controls

The Quant Loop terminal remains read-only by default. To expose the Research Control panel's write actions, launch it explicitly on localhost:

```bash
PYTHONPATH=src .venv/bin/python -m quant_loop_trader.dashboard.api --port 8787 --enable-controls
```

The control API accepts requests only from `127.0.0.1` / `::1`, even if the server is bound to a wider interface. Without `--enable-controls`, the panel stays visible but disabled and all existing dashboard GET endpoints remain read-only.

## BTC campaign fields

- **Market**: normally `BTCUSD`.
- **Horizon**: forward prediction horizon in days.
- **Experiment budget**: maximum experiments in the autonomy session.
- **Campaign ID**: explicit scientific campaign identity, for example `btc_2026_v1`.
- **Holdout start**: first date reserved from ordinary research.
- **Data end**: last date fetched for the campaign. It must be later than the holdout boundary so the final holdout data can exist while ordinary research remains filtered to pre-holdout observations.
- **Research starts**: comma-separated candidate training starts used by the autonomy grid.
- **Full validation**: runs the existing validation stack after each experiment.

Starting a campaign launches the existing `quant_loop_trader.autonomy` module in a child process with the corresponding `QLT_CRYPTO_*` environment settings. Output is appended to `data/logs/dashboard-control.log`; the normal dashboard polling continues to display experiments, validation, system state and activity.

The dashboard does **not** perform final holdout adjudication. That remains a separate one-shot scientific action so ordinary Command Center usage cannot accidentally consume the permanent holdout.

## Recommended 2026 BTC layout

For a genuinely untouched 2026 holdout, use:

```text
Campaign ID:     btc_2026_v1
Research starts: 2018-01-01, 2020-01-01, 2022-01-01
Holdout start:   2026-01-01
Data end:        current date
```

The model does not need calendar dates as predictive features. Timestamps remain in the research engine because chronological ordering, purging, replay windows and leakage checks depend on them.

## Launch model

The research-session launchd plist has no calendar trigger. Research starts only from the
localhost dashboard controls or an explicit supervisor/manual launch. The default session
budget is 100, which is deliberately larger than the normal candidate frontier; the
anti-duplicate `_already_run` check and campaign identity remain authoritative, so a
session naturally stops when there is no genuinely unexplored evidence left.
