# BTCUSD research workflow

Quant Loop supports BTCUSD as a first-class daily research market through the
Tiingo crypto endpoint. It uses the existing `TIINGO_API_KEY`; no separate
crypto credential is required.

## Scientific defaults

- Market symbol: `BTCUSD`
- Daily bars: Tiingo crypto `priceData`
- Calendar: 365 days/year
- Default research horizon: 5 days
- Default transaction cost: 5 bps per side
- Cost sensitivity: 5 / 10 / 25 / 50 bps per side
- Canonical h-day cost stress uses the **worst valid entry phase**, not only phase 0
- Final promotion must remain profitable at 25 bps per side
- Default crypto campaign: `btc_pre2024_v1`
- Permanent default campaign holdout begins `2024-01-01`
- Research memory is scoped by ticker + horizon, so SPY evidence does not alter
  BTCUSD duplicate counts or prior confidence.

The holdout boundary is campaign-level, not calculated independently for each
experiment. Changing an experiment's end date therefore cannot recycle a
previously hidden BTC observation into research.

A custom `QLT_CRYPTO_HOLDOUT_START` is rejected unless a new
`QLT_CRYPTO_CAMPAIGN_ID` is also supplied. Campaign identity and holdout boundary
are part of the experiment fingerprint, so changing the scientific window creates
a distinct experiment population rather than silently changing an old config.

## Data-integrity rule

BTCUSD is a 24/7 daily market. A Tiingo response is accepted only when it contains
**every requested UTC calendar day**, including the exact requested start and end.
A contiguous but truncated response fails closed and is never sealed as a valid
dataset snapshot.

Run the credentialed Tiingo smoke test before the first campaign on a new machine:

```bash
.venv/bin/python -m pytest -q -m integration \
  tests/test_connectors_integration.py -k tiingo_live_btcusd
```

The test skips if `TIINGO_API_KEY` is absent. A skip is not evidence that the live
Tiingo contract works.

## 1. Start Command Center

```bash
cd quant-loop-trader
PYTHONPATH=src .venv/bin/python -m quant_loop_trader.dashboard.api --port 8787
```

Open `http://127.0.0.1:8787`.

The market panel follows the selected experiment. When a BTCUSD experiment is
selected it reads `data/processed/BTCUSD.parquet` and uses the crypto calendar.

## 2. Run one controlled BTCUSD experiment

Do this before enabling a multi-experiment autonomous session:

```bash
.venv/bin/python -m quant_loop_trader.experiment \
  --ticker BTCUSD \
  --horizon 5 \
  --start 2018-01-01 \
  --end 2024-12-31
```

The run fetches BTCUSD through Tiingo Crypto, verifies exact daily coverage,
seals the dataset snapshot, excludes the permanent holdout from research, trains
baseline/improved models, and writes the same reproducible experiment bundle used
for equities.

## 3. Validate the experiment

Use the experiment ID printed by the previous command:

```bash
.venv/bin/python -m quant_loop_trader.experiment --validate <experiment_id>
```

Validation uses the same market-aware annualisation as the experiment and
Command Center. Walk-forward validation also recognises 24/7 data for legacy
callers that do not provide a ticker explicitly.

## 4. Start a single autonomous BTCUSD experiment

```bash
export QLT_AUTONOMOUS_ENABLED=true

.venv/bin/python -m quant_loop_trader.autonomy \
  --ticker BTCUSD \
  --horizon 5 \
  --max-experiments 1
```

Keep the first autonomous session at one experiment. If its dataset provenance,
sealed metrics, validation state, equity curve, drawdown and Command Center
market panel all reconcile, increase the budget in later sessions.

The bundled launchd research job is intentionally configured to the same
conservative BTCUSD / 5-day / one-experiment settings. Editing the plist does not
reload an already-installed LaunchAgent; unload/reload it after deploying changes.

## Starting a new, later BTC campaign

Do **not** move the default 2024 holdout while keeping the old campaign identity.
If old holdout data is going to become research data, that is a new scientific
campaign and it needs a new, later untouched holdout.

For example, a deliberately new 2026 campaign can be configured explicitly:

```bash
export QLT_CRYPTO_CAMPAIGN_ID=btc_2026_v1
export QLT_CRYPTO_HOLDOUT_START=2026-01-01
export QLT_CRYPTO_CAMPAIGN_STARTS=2018-01-01,2020-01-01,2022-01-01
export QLT_CRYPTO_CAMPAIGN_ENDS=2026-08-29
```

Only use a boundary whose final-holdout observations have not already been used
to tune or select the new campaign. The program can enforce campaign identity and
one-shot evidence; it cannot know what a human has previously inspected outside
the evidence store.

With those variables set, `autonomy --ticker BTCUSD` searches the explicitly
configured campaign ends rather than the frozen original grid.

## Holdout behavior

Experiments ending before the permanent campaign boundary can be researched and
validated, but final holdout adjudication refuses to consume a hidden period that
is not inside that experiment's dataset. A final-window experiment must contain
enough post-boundary observations before the one-shot holdout claim is taken.

For an h-day strategy, financial evaluation retains phase 0 for legacy chart
reconciliation but computes fee-stressed performance for every possible entry
phase. `cost_sensitivity_compounded` is the worst phase at each fee level. This
means the existing 25 bps promotion gate now fails if even one legitimate h-day
entry schedule is unprofitable.

The final promotion gate requires:

- holdout accuracy above the holdout majority-class base rate;
- statistical significance with the existing effective-sample rule;
- positive fully liquidated compounded phase-0 net return;
- liquidated strategy Sharpe at least as high as benchmark Sharpe; and
- positive **worst-phase** compounded return under the 25 bps-per-side cost stress.

Holdout evidence remains one-shot, sealed, hash-bound to the research bundle and
dataset snapshot, and atomically committed with the model lifecycle state.

## Trading boundary

Quant Loop is still a research and offline paper-simulation system. The repository
does not submit real broker orders. `paper_trading.py` deliberately contains only
an offline execution simulator. Do not treat a research champion as a live-trading
integration.