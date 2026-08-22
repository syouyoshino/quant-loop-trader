# Quant Loop Trader

Historical-replay stock prediction engine. The intended flow is:

1. Fetch and cache market and macro data in `data/raw/` and `data/cache/`.
2. Produce replay-ready feature data in `data/processed/`.
3. Run predictions strictly one historical timestamp at a time, using only data available at that timestamp.
4. Record results for evaluation before any paper-trading integration.

## Layout

```text
src/quant_loop_trader/  Application code
data/                   Local data root (market data is ignored by Git)
tests/                  Automated checks
.env                    Your local API credentials (ignored by Git)
.env.example            Safe credential template
```

## Local setup

Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy the safe template if needed, then paste your own values into `.env`:

```bash
cp .env.example .env
```

Keep `ALPACA_PAPER=true` until the strategy has been tested through historical replay and paper trading.
