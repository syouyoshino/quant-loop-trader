import { escape, kv, num, pct, pill, text, NA } from './util.js';

export function renderMarket(node, m) {
  if (!m || !m.available) {
    node.innerHTML = `<div class="panel-head"><h2>MARKET REGIME</h2></div>
      <div class="empty" style="height:120px">${escape(((m && m.reason) || 'NO MARKET DATA').toUpperCase())}</div>`;
    return;
  }
  const rv = m.realized_volatility || {};
  node.innerHTML = `<div class="panel-head"><h2>MARKET REGIME — ${escape(m.ticker)}</h2>
      <span class="dim">as of ${escape(m.as_of)} · ${m.rows} rows</span></div>
    <div class="cols-2">
      ${kv([
        ['CLOSE', num(m.close, 2)],
        ['VOLUME', text(m.volume)],
        ['1D', pct(m.change_1d)],
        ['30D', pct(m.change_30d)],
        ['TREND', pill(m.trend)],
        ['SMA50 / SMA200', `${num(m.sma50, 1)} / ${num(m.sma200, 1)}`],
      ])}
      ${kv([
        ['RV 7D', pct(rv['7d'])],
        ['RV 30D', pct(rv['30d'])],
        ['RV 90D', pct(rv['90d'])],
        [`RV ${m.calendar_days}D`, pct(rv[`${m.calendar_days}d`])],
        ['VOL REGIME', pill(m.volatility_regime)],
        ['VOL PERCENTILE', m.volatility_percentile === null ? NA : (m.volatility_percentile * 100).toFixed(0) + 'th'],
      ])}
    </div>
    <div class="note">computed from ${escape(m.source)} (research snapshot, not a live feed).
      Not implemented for this market: ${escape((m.unavailable_fields || []).join(', '))} —
      ${escape(m.unavailable_reason || '')}</div>`;
}
