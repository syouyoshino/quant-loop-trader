import { bar, duration, int, pill, stamp, text, NA, escape } from './util.js';

// The control-room block: what the lab is doing right now, above returns.
export function renderCycle(node, cycle, progress) {
  if (!cycle || cycle.status === 'NO_DATA') {
    node.innerHTML = `<div class="panel-head"><h2>CURRENT RESEARCH CYCLE</h2></div>
      <div class="empty" style="height:90px">NO AUTONOMY SESSION RECORDED</div>
      <div class="note">${escape((cycle && cycle.note) || 'data/logs/session.log holds no session summary')}</div>`;
    return;
  }
  const done = cycle.completed_experiments;
  const planned = cycle.planned_experiments;
  const frac = (typeof done === 'number' && typeof planned === 'number' && planned > 0)
    ? done / planned : null;

  const left = [
    ['STATUS', pill(cycle.status)],
    ['MODE', text(cycle.mode)],
    ['STARTED', stamp(cycle.started_at)],
    ['COMPLETED', cycle.completed_at ? stamp(cycle.completed_at) : NA],
    ['ELAPSED', duration(cycle.elapsed_s)],
  ];
  const mid = [
    ['ACTIVE', text(cycle.active_experiment || cycle.last_experiment)],
    ['MARKET', text(cycle.active_market)],
    ['HORIZON', cycle.active_horizon ? cycle.active_horizon + 'd' : NA],
    ['STAGE', text(cycle.active_stage)],
    ['NEXT', cycle.next_experiment ? text(cycle.next_experiment)
      : `<span class="na">N/A</span>`],
  ];
  const right = [
    ['CYCLES DONE', int(progress.cycles_completed)],
    ['EXPERIMENTS', int(progress.experiments_completed)],
    ['ACTIVE', int(progress.experiments_active)],
    ['QUEUED (GRID)', progress.experiments_queued === null ? NA : int(progress.experiments_queued)],
    ['HYPOTHESES', int(progress.hypotheses_researched)],
  ];

  node.innerHTML = `
    <div class="cycle-top">
      <div class="cycle-no"><small>CYCLE</small>${cycle.cycle_number === null || cycle.cycle_number === undefined
        ? '—' : String(cycle.cycle_number).padStart(3, '0')}</div>
      <div>${bar(frac, 22)}</div>
      <div class="cycle-pct">${frac === null ? '<span class="na">N/A</span>'
        : `<span class="${frac >= 1 ? 'pos' : 'warn'}">${Math.round(frac * 100)}%</span>`}</div>
      <div class="cycle-count">${done === null || done === undefined ? 'N/A' : done} /
        ${planned === null || planned === undefined ? 'N/A' : planned} EXPERIMENTS</div>
      <div class="dim">${escape(cycle.note || '')}</div>
    </div>
    <div class="cycle-tallies">
      ${tally('PASSED', cycle.keeps, 'pos')}
      ${tally('IMPROVE', cycle.improves, 'warn')}
      ${tally('REJECTED', cycle.rejects, 'neg')}
      ${tally('VALIDATED', cycle.validation_passes, 'pos')}
      ${tally('VALIDATION FAILED', cycle.validation_failures, 'neg')}
      ${tally('FAILED RUNS', cycle.failed_experiments, 'neg')}
      ${tally('CHAMPIONS', progress.champions, 'pos')}
      ${tally('GRID REMAINING', cycle.grid_remaining, 'info')}
    </div>
    <div class="cycle-grid">
      ${col(left)}${col(mid)}${col(right)}
      <div>${row('HYPOTHESIS', hypothesisLine(cycle.active_hypothesis))}</div>
    </div>`;
}

function tally(k, v, cls) {
  const shown = (v === null || v === undefined) ? '<span class="na">N/A</span>'
    : `<span class="${v ? cls : 'dim'}">${v}</span>`;
  return `<div class="tally"><span class="k">${k}</span><span class="v">${shown}</span></div>`;
}
function col(rows) { return `<div>${rows.map(([k, v]) => row(k, v)).join('')}</div>`; }
function row(k, v) { return `<div class="row"><span class="k">${k}</span><span class="v">${v}</span></div>`; }
function hypothesisLine(h) {
  return h ? `<span title="${escape(h)}">${escape(h.slice(0, 64))}${h.length > 64 ? '…' : ''}</span>` : NA;
}
