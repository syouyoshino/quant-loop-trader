import { el, pill, text, utcClock, ago, NA } from './util.js';

export function renderHeader(sys, markets) {
  const f = el('topbar-fields');
  if (!f) return;
  const data = sys.data || {};
  const db = sys.database || {};
  const repo = sys.repository || {};
  const fields = [
    ['MKT', markets && markets.length ? markets.join('/') : NA],
    ['AUTONOMY', pill(sys.autonomy)],
    ['MODE', text(sys.mode)],
    ['CYCLE', sys.cycle === null || sys.cycle === undefined ? NA : String(sys.cycle).padStart(3, '0')],
    ['DATA', pill(data.status)],
    ['DB', pill(db.status)],
    ['HEARTBEAT', sys.last_heartbeat ? ago(sys.last_heartbeat) : NA],
    ['REPO', repo.commit ? text(repo.commit) + (repo.dirty ? '<span class="warn">*</span>' : '') : NA],
    ['ERR', sys.errors.length ? `<span class="neg">${sys.errors.length}</span>` : '0'],
    ['WARN', sys.warnings.length ? `<span class="warn">${sys.warnings.length}</span>` : '0'],
  ];
  f.innerHTML = fields.map(([k, v]) => `<span class="kv"><span class="k">${k}</span>${v}</span>`).join('');
}

export function startClock() {
  const tick = () => { const c = el('clock'); if (c) c.textContent = utcClock(); };
  tick();
  setInterval(tick, 1000);
}
