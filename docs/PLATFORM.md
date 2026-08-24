# Quant Loop Trader — Research Platform

An autonomous quantitative research laboratory. It predicts, tests against hidden
futures, autopsies failures, improves, retests, and remembers. It is **not** a
trading bot: nothing here places real orders, and autonomous research is disabled
until explicitly activated.

## Architecture

```
connectors/          PIT data acquisition (Tiingo, Alpaca historical, FRED, SEC EDGAR)
data.py              market data fetch/cache/checksums + DuckDB migrations
replay.py            ReplayEngine.get_snapshot (available_time <= T) + evaluate_future + pit_filter
features/
  technical.py       returns/momentum/volatility/MA/RSI (shift(1) discipline)
  macro.py           rates/inflation/unemployment/regime via as-of publication joins
  fundamental.py     SEC XBRL growth/margin/quality via exact filing-date PIT
  pit.py             as-of join primitive
models/
  prediction.py      frozen Prediction objects (content-hashed)
  registry.py        versioned model interface: random/majority/logistic/xgboost
experiment.py        run_experiment / run_horizons / registry API / build_train_test
evaluation.py        time-split(purge), accuracy/Brier, Sharpe/Sortino/Calmar/VaR/ES, trade quality
validation/
  holdout.py         permanently hidden out-of-time segment
  walkforward.py     expanding-window folds
  ablation.py        feature-group ablation
  multiple_testing.py  BH-FDR + deflated Sharpe Ratio
  dashboard.py       bias/mining/tamper sentinels
agents.py            statistical/adversarial/replication validation gate + champion promotion
research/            hypothesis engine → novelty check → ranking → experiment configs
portfolio/construction.py  sizing schemes, position caps, drawdown stop
strategies/framework.py    Strategy ABC (PIT-only contract)
automation/          DuckDB task queue + ResearchController (inert by default)
monitoring/          heartbeat, health aggregation, provider-agnostic alerts
paper_trading.py     Order/simulator/tracker (offline; broker double-keyed off)
```

## Data flow

1. `fetch_ohlcv` → cached Parquet (`event_time`, `available_time` per row).
2. `ReplayEngine.get_snapshot(ticker, T)` → only rows with `available_time <= T`.
3. `build_train_test` → labels (`close[t+h]/close[t]`), features, purge boundary,
   time-ordered split. Creator and all reviewers share this one pipeline.
4. Train → predict → freeze artifacts → SHA-256 `predictions.lock`.
5. `evaluate()` reveals outcomes (only evaluation touches the future).
6. Autopsy by volatility regime → improvement hypothesis → retest on identical
   hidden test.
7. Validation gate: statistical (majority-class null) + adversarial
   (feature-shuffle, label-randomisation, regime concentration) + independent
   replication. Champions are promoted **only** on APPROVED.
8. Outcomes written to research memory with confidence updates.

## Validation process

A strategy is acceptable only if it:
1. has no future leakage (PIT joins, purge, truncation-invariance tests),
2. beats the majority-class base rate with significance,
3. survives label-randomisation and feature-shuffle null tests,
4. survives transaction costs,
5. passes risk analysis (drawdown/VaR/ES),
6. beats buy-and-hold,
7. replicates independently from documented config alone,
8. survives multiple-testing correction (BH-FDR, deflated Sharpe).

## Safety boundaries

- No real-money trading anywhere in the codebase. Paper broker is offline-only.
- Autonomous controller requires `ResearchController(enabled=True)` AND env
  `QLT_AUTONOMOUS_ENABLED=true`. Tests prove inertness.
- Paper broker additionally requires `allow=True` AND `QLT_PAPER_ENABLED=true`.
- Evaluation rules live in immutable modules; agents may propose but never weaken.
- Failed experiments are permanent records — deletion is not a code path.

## Research lifecycle

hypothesis → novelty check → priority rank → experiment → lock → hidden test →
autopsy → improve/reject → validation gate → memory/belief update → repeat.
Hypotheses whose families fail repeatedly are skipped automatically.

## Activation process

1. Refresh hypothesis frontier (engine generates candidates automatically).
2. Wire workers into `automation.controller.WORKERS`.
3. Set `QLT_AUTONOMOUS_ENABLED=true` and start the controller with `enabled=True`.
4. Monitor via `python -m quant_loop_trader.monitoring.health` and heartbeat age;
   set `ALERT_WEBHOOK_URL` for push notifications.

## Operations

```bash
.venv/bin/python -m pytest -q -m "not integration"   # full offline suite
launchctl list | grep quantloop                      # scheduled sessions
cat data/logs/heartbeat.json                         # liveness
```
