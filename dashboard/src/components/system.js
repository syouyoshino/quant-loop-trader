import { ago, escape, int, kv, pill, stamp, text, NA } from './util.js';

// An unsealed directory proves a process started, not that one is alive.
function activeRuns(a) {
  if (!a) return NA;
  const parts = [`${int(a.running)} running`];
  if (a.stale) parts.push(`<span class="warn">${a.stale} stale</span>`);
  if (a.orphaned) parts.push(`<span class="neg">${a.orphaned} orphaned</span>`);
  return parts.join(' · ');
}

export function renderSystem(node, sys) {
  const q = sys.queue || {};
  const data = sys.data || {};
  const ds = (data.datasets || [])[0];
  const rows = [
    ['AUTONOMY', pill(sys.autonomy)],
    ['ENABLED FLAG', sys.autonomy_enabled === null ? NA : (sys.autonomy_enabled ? 'true' : 'false')],
    ['CYCLE', sys.cycle === null || sys.cycle === undefined ? NA : String(sys.cycle).padStart(3, '0')],
    ['EXPERIMENT', text(sys.experiment)],
    ['STAGE', text(sys.stage)],
    ['PROGRESS', (sys.progress && sys.progress.completed !== null && sys.progress.completed !== undefined)
      ? `${sys.progress.completed} / ${sys.progress.planned === null || sys.progress.planned === undefined
        ? 'N/A' : sys.progress.planned}` : NA],
    ['ACTIVE RUNS', activeRuns(sys.active_runs)],
    ['GRID REMAINING', q.grid_remaining === null || q.grid_remaining === undefined ? NA : int(q.grid_remaining)],
    ['TASK QUEUE', `${int(q.tasks_pending)} pending / ${int(q.tasks_failed)} failed`],
    ['LAST HEARTBEAT', sys.last_heartbeat ? `${ago(sys.last_heartbeat)} ago` : NA],
    ['LAST EXPERIMENT', sys.last_experiment_completed ? stamp(sys.last_experiment_completed) : NA],
    ['EXPERIMENTS (AUTHORITATIVE)', int(sys.experiments_total)],
    ['DATA', `${pill(data.status)} ${ds ? `${ds.ticker} → ${ds.last_event_time} (${ds.age_days}d)` : ''}`],
    ['DATABASE', pill((sys.database || {}).status)],
    ['REPO', (sys.repository || {}).commit
      ? text(sys.repository.commit) + (sys.repository.dirty ? ' <span class="warn">dirty</span>' : ' clean') : NA],
  ];
  const jobs = (sys.scheduled_jobs || []).map((j) =>
    `<div class="row"><span class="k">${escape(j.label)}</span>
      <span class="v">${j.loaded === null ? NA : j.loaded ? '<span class="pos">loaded</span>'
        : '<span class="na">not loaded</span>'}</span></div>`).join('');

  node.innerHTML = `<div class="panel-head"><h2>SYSTEM / AUTONOMY</h2>
      <span class="dim">${stamp(sys.server_time)}</span></div>
    ${kv(rows)}
    <div class="rule"></div>
    <div class="kv-list">${jobs || '<span class="dim">no launchd jobs declared</span>'}</div>
    ${sys.errors.length ? `<div class="rule"></div>${sys.errors.map((e) =>
      `<div class="neg">ERR ${escape(e)}</div>`).join('')}` : ''}
    ${sys.warnings.length ? sys.warnings.map((w) =>
      `<div class="warn">WRN ${escape(w)}</div>`).join('') : ''}`;
}
