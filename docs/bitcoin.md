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
- Final promotion must remain profitable at 25 bps per side
- Permanent crypto campaign holdout begins `2024-01-01` by default
- Override the permanent boundary with `QLT_CRYPTO_HOLDOUT_START=YYYY-MM-DD`
- Research memory is scoped by ticker + horizon, so SPY evidence does not alter
  BTCUSD duplicate counts or prior confidence.

The holdout boundary is campaign-level, not calculated independently for each
experiment. Changing an experiment's end date therefore cannot recycle a
previously hidden BTC observation into research.

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

The run fetches BTCUSD through Tiingo Crypto, verifies that the daily series has
no missing calendar days, seals the dataset snapshot, excludes the permanent
holdout from research, trains baseline/improved models, and writes the same
reproducible experiment bundle used for equities.

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

## Holdout behavior

Experiments ending before the permanent crypto boundary can be researched and
validated, but final holdout adjudication refuses to consume a hidden period
that is not inside that experiment's dataset. A final-window experiment must
contain enough post-boundary observations before the one-shot holdout claim is
taken.

The final promotion gate requires:

- holdout accuracy above the holdout majority-class base rate;
- statistical significance with the existing effective-sample rule;
- positive fully liquidated compounded net return;
- liquidated strategy Sharpe at least as high as benchmark Sharpe; and
- positive compounded return under the 25 bps-per-side cost stress.

Holdout evidence remains one-shot, sealed, hash-bound to the research bundle and
dataset snapshot, and atomically committed with the model lifecycle state.
