import { escape, num, pct, pill, ratio, text, NA } from './util.js';

export function renderChampions(node, data, onSelect) {
  const rows = data.champions.length ? data.champions : data.eligible;
  const heading = data.champions.length ? 'CHAMPIONS' : 'CHAMPIONS — NONE · SHOWING ELIGIBLE';
  node.innerHTML = `
    <div class="panel-head"><h2>${heading}</h2>
      <span class="dim">${Object.entries(data.counts).map(([k, v]) => `${k}:${v}`).join(' · ')}</span></div>
    ${rows.length ? `<div class="scroll-x"><table>
      <thead><tr><th class="left">ID</th><th>STATE</th><th>NET<sup>r</sup></th><th>EXCESS<sup>r</sup></th><th>SHARPE<sup>r</sup></th>
        <th>SORTINO</th><th>CALMAR</th><th>MAX DD</th><th>VOL</th><th>EDGE</th></tr></thead>
      <tbody>${rows.map((c) => `
        <tr data-id="${escape(c.experiment_id)}">
          <td class="left">${escape(c.experiment_id)}</td>
          <td>${pill(c.lifecycle)}</td>
          <td>${pct(c.net_return)}</td><td>${pct(c.excess_return)}</td>
          <td>${ratio(c.sharpe)}</td><td>${ratio(c.sortino)}</td><td>${ratio(c.calmar)}</td>
          <td>${pct(c.max_drawdown)}</td><td>${pct(c.annualized_volatility)}</td>
          <td>${pill((c.edge || {}).status)}</td>
        </tr>
        <tr><td class="left dim" colspan="10">evidence: ${Object.entries(c.evidence || {})
          .map(([k, v]) => `${k}=${v}`).join(' · ')}</td></tr>
        <tr><td class="left dim" colspan="10">holdout: ${holdout(c.holdout)}</td></tr>`).join('')}
      </tbody></table></div>` : '<div class="empty" style="height:80px">NO CHAMPION OR ELIGIBLE MODEL</div>'}
    <div class="note">ʳ research test window (sealed metrics.json) — not the hidden holdout</div>
    ${data.note ? `<div class="note">${escape(data.note)}</div>` : ''}
    <div class="rule"></div>
    <div class="dim">RETURN CORRELATION</div>
    ${correlation(data.correlation)}`;
  node.querySelectorAll('tbody tr[data-id]').forEach((tr) =>
    tr.addEventListener('click', () => onSelect(tr.dataset.id)));
}

// Row metrics are the sealed research-window evaluate() output; the hidden
// holdout persists only its economic gate, so the rest stays N/A.
function holdout(h) {
  if (!h || !h.available) return 'NOT RUN';
  return [
    pill(h.status),
    `net ${pct(h.net_return)}`,
    `sharpe ${ratio(h.sharpe)} vs bench ${ratio(h.benchmark_sharpe)}`,
    `acc ${pct(h.accuracy)} vs base ${pct(h.base_rate)}`,
    `n=${text(h.n)}`,
    `max dd ${NA} · vol ${NA} (not persisted)`,
  ].join(' · ');
}

function correlation(c) {
  if (!c || !c.available) {
    return `<div class="dim">N/A — ${escape((c && c.reason) || 'no data')}</div>`;
  }
  return `<div class="scroll-x"><table>
    <thead><tr><th class="left"></th>${c.ids.map((i) => `<th>${escape(i.slice(-8))}</th>`).join('')}</tr></thead>
    <tbody>${c.matrix.map((row, i) => `<tr><td class="left">${escape(c.ids[i].slice(-8))}</td>
      ${row.map((v) => `<td class="${v !== null && v > 0.8 ? 'warn' : ''}">${num(v, 2)}</td>`).join('')}</tr>`).join('')}
    </tbody></table></div>
    ${c.highly_correlated.length ? c.highly_correlated.map(([a, b, v]) =>
      `<div class="warn">⚠ ${escape(a)} ↔ ${escape(b)} ρ=${num(v, 2)} — effectively the same trade</div>`).join('') : ''}`;
}
