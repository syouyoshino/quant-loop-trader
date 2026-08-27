// Formatting. A missing value is ALWAYS N/A — never zero, never a guess.
export const NA = '<span class="na">N/A</span>';

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

export function num(v, dp = 2) { return isNum(v) ? v.toFixed(dp) : NA; }

export function pct(v, dp = 2) {
  if (!isNum(v)) return NA;
  return `<span class="${v > 0 ? 'pos' : v < 0 ? 'neg' : ''}">${(v * 100).toFixed(dp)}%</span>`;
}

export function pctPlain(v, dp = 2) { return isNum(v) ? (v * 100).toFixed(dp) + '%' : NA; }

export function signed(v, dp = 2) {
  if (!isNum(v)) return NA;
  return `<span class="${v > 0 ? 'pos' : v < 0 ? 'neg' : ''}">${v > 0 ? '+' : ''}${v.toFixed(dp)}</span>`;
}

export function ratio(v, dp = 2) {
  if (!isNum(v)) return NA;
  return `<span class="${v > 0 ? 'pos' : v < 0 ? 'neg' : ''}">${v.toFixed(dp)}</span>`;
}

export function int(v) { return isNum(v) ? String(Math.round(v)) : NA; }

export function text(v) {
  if (v === null || v === undefined || v === '') return NA;
  return escape(String(v));
}

export function escape(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

export function duration(seconds) {
  if (!isNum(seconds)) return NA;
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

export function ago(iso) {
  if (!iso) return NA;
  const d = (Date.now() - Date.parse(iso)) / 1000;
  if (!Number.isFinite(d)) return NA;
  if (d < 60) return `${Math.round(d)}s`;
  if (d < 3600) return `${Math.round(d / 60)}m`;
  if (d < 86400) return `${Math.round(d / 3600)}h`;
  return `${Math.round(d / 86400)}d`;
}

export function stamp(iso, withTime = true) {
  if (!iso) return NA;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return escape(iso);
  const date = d.toISOString().slice(0, 10);
  return withTime ? `${date} ${d.toISOString().slice(11, 19)}` : date;
}

export function utcClock(d = new Date()) {
  const M = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];
  return `${String(d.getUTCDate()).padStart(2, '0')} ${M[d.getUTCMonth()]} ${d.getUTCFullYear()} `
    + `${d.toISOString().slice(11, 19)} UTC`;
}

const STATUS_CLASS = {
  PASS: 'pass', APPROVED: 'pass', KEEP: 'pass', RUNNING: 'warn', CURRENT: 'warn',
  IMPROVE: 'warn', STALLED: 'fail', FAIL: 'fail', REJECTED: 'fail', REJECT: 'fail',
  CRASHED: 'fail', NOT_RUN: 'off', NOT_AVAILABLE: 'off', UNKNOWN: 'off', IDLE: 'info',
  DISABLED: 'off', COMPLETED: 'info', OK: 'pass', STALE: 'warn', ERROR: 'fail',
  UNAVAILABLE: 'fail', STABLE: 'pass', WEAKENING: 'warn', SEVERE_DECAY: 'fail',
  LOW: 'info', NORMAL: 'info', HIGH: 'warn', EXTREME: 'fail', UP: 'pass', DOWN: 'neg',
};

export function pill(status) {
  if (!status) return NA;
  const cls = STATUS_CLASS[status] || 'off';
  return `<span class="pill ${cls}">${escape(String(status).replace(/_/g, ' '))}</span>`;
}

export const MARK = { PASS: '✓', FAIL: '×', CURRENT: '●', NOT_RUN: '○', NOT_AVAILABLE: '·' };

export function bar(fraction, width = 20) {
  if (!isNum(fraction)) return `<span class="bar"><span class="rest">${'░'.repeat(width)}</span></span>`;
  const f = Math.max(0, Math.min(1, fraction));
  const on = Math.round(f * width);
  return `<span class="bar"><span class="fill">${'█'.repeat(on)}</span>`
    + `<span class="rest">${'░'.repeat(width - on)}</span></span>`;
}

export function kv(rows) {
  return `<div class="kv-list">${rows.map(([k, v]) =>
    `<div class="row"><span class="k">${escape(k)}</span><span class="v">${v}</span></div>`).join('')}</div>`;
}

export function el(id) { return document.getElementById(id); }
