import { escape, num, pct, pill, ratio, stamp, duration, text, NA } from './util.js';

const COLUMNS = [
  ['id', 'ID', 'left', (r) => `<span title="${escape(r.id)}">${escape(r.id)}</span>`],
  ['cycle', 'CYC', '', (r) => (r.cycle === null ? '<span class="na">—</span>' : String(r.cycle).padStart(3, '0'))],
  ['market', 'MKT', 'left', (r) => text(r.market)],
  ['hypothesis', 'HYPOTHESIS', 'left', (r) => `<span class="truncate" title="${escape(r.hypothesis || '')}">${escape((r.hypothesis || 'N/A').slice(0, 46))}</span>`],
  ['feature_family', 'FEATURES', 'left', (r) => text(r.feature_family)],
  ['horizon', 'H', '', (r) => (r.horizon ? r.horizon + 'd' : NA)],
  ['model', 'MODEL', 'left', (r) => text(r.model)],
  ['stage', 'STAGE', 'left', (r) => text(r.stage)],
  ['status', 'STATUS', '', (r) => pill(r.status)],
  ['net_return', 'NET', '', (r) => pct(r.net_return)],
  ['excess_return', 'EXCESS', '', (r) => pct(r.excess_return)],
  ['sharpe', 'SHARPE', '', (r) => ratio(r.sharpe)],
  ['max_drawdown', 'MAX DD', '', (r) => pct(r.max_drawdown)],
  ['volatility', 'VOL', '', (r) => num(r.volatility, 4)],
  ['p_value', 'p', '', (r) => num(r.p_value, 4)],
  ['dsr', 'DSR', '', (r) => num(r.dsr, 3)],
  ['fdr', 'FDR', '', (r) => pill(r.fdr)],
  ['validation', 'VALID', '', (r) => pill(r.validation)],
  ['holdout', 'HOLDOUT', '', (r) => pill(r.holdout)],
  ['started', 'STARTED', '', (r) => stamp(r.started)],
  ['duration_s', 'DUR', '', (r) => duration(r.duration_s)],
];

let sortKey = 'started';
let sortDir = -1;

export function renderTable(node, payload, state, onSelect) {
  const filters = payload.filters || {};
  const rows = [...payload.experiments].sort(compare);
  node.innerHTML = `
    <div class="panel-head">
      <h2>EXPERIMENTS <span class="dim">${payload.total} shown${
        (payload.population || {}).quarantined ? ` · ${payload.population.quarantined} quarantined` : ''}</span></h2>
      <div class="filters">
        ${select('market', 'MARKET', filters.markets, state.filters.market)}
        ${select('cycle', 'CYCLE', filters.cycles, state.filters.cycle)}
        ${select('status', 'STATUS', filters.statuses, state.filters.status)}
        ${select('stage', 'STAGE', filters.stages, state.filters.stage)}
        <input type="search" data-filter="hypothesis" placeholder="hypothesis…"
          value="${escape(state.filters.hypothesis || '')}" size="14">
        <input type="date" data-filter="from" value="${escape((state.filters.from || '').slice(0, 10))}">
        <input type="date" data-filter="to" value="${escape((state.filters.to || '').slice(0, 10))}">
        <label class="dim"><input type="checkbox" data-filter="champion_only"
          ${state.filters.champion_only ? 'checked' : ''}> CHAMPION ONLY</label>
        <label class="dim" title="quarantined runs predate the current pipeline"><input type="checkbox"
          data-filter="include_quarantined"
          ${state.filters.include_quarantined ? 'checked' : ''}> INCLUDE QUARANTINED</label>
      </div>
    </div>
    <div class="scroll-y scroll-x"><table>
      <thead><tr>${COLUMNS.map(([key, label, cls]) =>
        `<th class="${cls}" data-sort="${key}">${label}${sortKey === key ? (sortDir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr></thead>
      <tbody>${rows.length ? rows.map((r) => `
        <tr data-id="${escape(r.id)}" class="${r.id === state.selected ? 'selected' : ''}">
          ${COLUMNS.map(([, , cls, fmt]) => `<td class="${cls}">${fmt(r)}</td>`).join('')}
        </tr>`).join('') : `<tr><td colspan="${COLUMNS.length}" class="left dim">NO EXPERIMENTS MATCH</td></tr>`}
      </tbody></table></div>`;

  node.querySelectorAll('th[data-sort]').forEach((th) => th.addEventListener('click', () => {
    const key = th.dataset.sort;
    if (key === sortKey) sortDir = -sortDir; else { sortKey = key; sortDir = -1; }
    renderTable(node, payload, state, onSelect);
  }));
  node.querySelectorAll('tbody tr[data-id]').forEach((tr) => tr.addEventListener('click', () => {
    onSelect(tr.dataset.id);
  }));
  node.querySelectorAll('[data-filter], select[data-filter-key]').forEach((input) => {
    const key = input.dataset.filter || input.dataset.filterKey;
    const event = input.tagName === 'INPUT' && input.type === 'search' ? 'change' : 'change';
    input.addEventListener(event, () => {
      state.filters[key] = input.type === 'checkbox' ? input.checked : input.value;
      document.dispatchEvent(new CustomEvent('terminal:filters'));
    });
  });
}

function select(key, label, options, value) {
  if (!options || !options.length) return '';
  return `<select data-filter-key="${key}">
    <option value="">${label}: ALL</option>
    ${options.map((o) => `<option value="${escape(o)}" ${String(o) === String(value) ? 'selected' : ''}>${escape(o)}</option>`).join('')}
  </select>`;
}

function compare(a, b) {
  const x = a[sortKey], y = b[sortKey];
  if (x === y) return 0;
  if (x === null || x === undefined) return 1;
  if (y === null || y === undefined) return -1;
  return (x > y ? 1 : -1) * sortDir;
}
