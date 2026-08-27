import { kv, pct, ratio, int, text, NA, escape } from './util.js';

export function renderRisk(node, risk) {
  if (!risk || !risk.available) {
    node.innerHTML = `<div class="panel-head"><h2>RISK</h2></div>
      <div class="empty" style="height:180px">${escape(((risk && risk.reason) || 'NO DATA').toUpperCase())}</div>`;
    return;
  }
  const vol = (risk.realized_volatility || []).map((w) => [
    `RV ${w.label}${w.span_days !== w.requested_days ? ` (${w.span_days}d)` : ''}`,
    w.available ? pct(w.value) : NA,
  ]);
  const dd = risk.max_drawdown_period || {};
  node.innerHTML = `<div class="panel-head"><h2>RISK / VOLATILITY</h2>
      <span class="dim">${int(risk.calendar_days)} d/yr</span></div>
    <div class="cols-2">
      ${kv([...vol,
        ['ANN. VOL', pct(risk.annualized_volatility)],
        ['DOWNSIDE VOL', pct(risk.annualized_downside_volatility)],
      ])}
      ${kv([
        ['VaR 95 (bucket)', pct(risk.var_95)],
        ['CVaR 95 (bucket)', pct(risk.expected_shortfall_95)],
        ['LARGEST LOSS', pct(risk.largest_loss)],
        ['WORST PERIOD', risk.worst_period ? `${text(risk.worst_period.t)} ${pct(risk.worst_period.return)}` : NA],
        ['CURRENT DD', pct(risk.current_drawdown)],
        ['MAX DD', pct(risk.max_drawdown)],
      ])}
    </div>
    <div class="rule"></div>
    ${kv([
      ['MAX DD PERIOD', `${text(dd.peak)} → ${text(dd.trough)}${dd.recovered ? ` → ${dd.recovered}` : ' (unrecovered)'}`],
      ['DAYS UNDERWATER (NOW)', int(risk.days_underwater_current)],
      ['LONGEST UNDERWATER', `${int(risk.days_underwater_longest)}d`],
      ['LONGEST RECOVERY', `${int(risk.longest_recovery_days)}d`],
    ])}
    <div class="note">${escape(risk.note)}</div>`;
}

export function renderEdge(node, edge, rolling) {
  if (!edge) { node.innerHTML = ''; return; }
  const item = (k, v) => `<div class="item"><span class="k">${k}</span><span class="v">${v}</span></div>`;
  const cls = { STABLE: 'pos', WEAKENING: 'warn', SEVERE_DECAY: 'neg' }[edge.status] || 'na';
  node.innerHTML = `
    ${item('EDGE STATUS', `<span class="${cls}">${escape((edge.status || 'N/A').replace('_', ' '))}</span>`)}
    ${item('RECENT SHARPE', ratio(edge.recent_sharpe))}
    ${item('HISTORICAL SHARPE', ratio(edge.historical_sharpe))}
    ${item('RATIO', ratio(edge.sharpe_ratio))}
    ${item('RECENT EXCESS', pct(edge.recent_excess_return))}
    ${item('HISTORICAL EXCESS', pct(edge.historical_excess_return))}
    ${item('WINDOW', edge.window_buckets ? `${edge.window_buckets} buckets` : NA)}
    <div class="item" style="flex:1"><span class="k">RULE</span>
      <span class="dim">${escape(edge.rule || edge.reason || '')}</span></div>`;
}
