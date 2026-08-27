import { kv, pct, ratio, int, num, text, NA, escape } from './util.js';

// Every field is a sealed metric or a value recomputed from the same series.
export function renderMetrics(node, perf, baseline) {
  if (!perf) { node.innerHTML = '<div class="empty" style="height:200px">NO EXPERIMENT</div>'; return; }
  const left = [
    ['NET RETURN', pct(perf.net_return)],
    ['BENCHMARK', pct(perf.benchmark_return)],
    ['EXCESS', pct(perf.excess_return)],
    ['CAGR', pct(perf.cagr)],
    ['BENCH CAGR', pct(perf.benchmark_cagr)],
    ['SHARPE', ratio(perf.sharpe)],
    ['SHARPE (BM)', ratio(perf.sharpe_benchmark)],
    ['SORTINO', ratio(perf.sortino)],
    ['CALMAR', ratio(perf.calmar)],
    ['MAX DD', pct(perf.max_drawdown)],
    ['CURRENT DD', pct(perf.current_drawdown)],
  ];
  const right = [
    ['ANN. VOL', pct(perf.annualized_volatility)],
    ['DOWNSIDE VOL', pct(perf.annualized_downside_volatility)],
    ['WIN RATE', pct(perf.win_rate, 1)],
    ['PROFIT FACTOR', ratio(perf.profit_factor)],
    ['EXPECTANCY', pct(perf.expectancy, 3)],
    ['EXPOSURE', pct(perf.exposure, 1)],
    ['TRADES', int(perf.n_trades)],
    ['TURNOVER', num(perf.turnover, 4)],
    ['COST DRAG', pct(perf.cost_drag, 3)],
    ['ACCURACY', pct(perf.accuracy, 2)],
    ['p-VALUE', num(perf.p_value, 4)],
  ];
  const ci = Array.isArray(perf.return_ci95)
    ? `[${(perf.return_ci95[0] * 100).toFixed(3)}%, ${(perf.return_ci95[1] * 100).toFixed(3)}%]` : NA;

  node.innerHTML = `<div class="panel-head"><h2>PERFORMANCE</h2>
      <span class="dim">${escape(perf.variant.toUpperCase())} · ${text(perf.ticker)} ${perf.horizon}d</span></div>
    <div class="cols-2">${kv(left)}${kv(right)}</div>
    <div class="rule"></div>
    ${kv([
      ['MEAN BUCKET RETURN CI95', ci],
      ['RETURN BUCKETS', int(perf.n_return_buckets)],
      ['TEST OBSERVATIONS', int(perf.n_test)],
      ['BASELINE SHARPE', baseline ? ratio(baseline.sharpe) : NA],
      ['BASELINE NET RETURN', baseline ? pct(baseline.net_return) : NA],
    ])}
    <div class="note">annualised on ${int(perf.calendar_days)} d/yr ÷ ${perf.horizon}d buckets
      = ${num(perf.periods_per_year, 1)} periods/yr</div>`;
}
