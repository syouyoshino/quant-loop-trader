import { escape, num, pill, text, NA } from './util.js';

// Individual tests stay visible: no single "AI confidence" number.
export function renderValidation(node, v) {
  if (!v) { node.innerHTML = '<div class="empty" style="height:160px">NO EXPERIMENT</div>'; return; }
  const t = v.tests;
  const rows = [
    ['Statistical significance', t.significance.status, `p=${num(t.significance.p_value, 4)}`],
    ['Effective sample size', t.significance.n_effective === null ? 'NOT_AVAILABLE' : 'PASS',
      `${text(t.significance.n_effective)} buckets / ${text(t.significance.n_test)} obs`],
    ['Replication', t.replication.status, (t.replication.tests || []).join(', ')],
    ['Walk-forward', t.walk_forward.status,
      t.walk_forward.mean_accuracy === null ? '' :
        `mean acc ${num(t.walk_forward.mean_accuracy, 4)} ± ${num(t.walk_forward.dispersion, 4)} · ${(t.walk_forward.folds || []).length} folds`],
    ['Adversarial', t.adversarial.status, (t.adversarial.issues || []).join(' · ')],
    ['Ablation', t.ablation.status, t.ablation.variants ? `${t.ablation.variants.length} variants` : ''],
    ['DSR', t.dsr.status, t.dsr.dsr === null ? '' : `DSR ${num(t.dsr.dsr, 3)} · ${text(t.dsr.verdict)} · ${text(t.dsr.n_trials)} trials`],
    ['FDR', t.fdr.status, t.fdr.detail || ''],
    ['Hidden holdout', t.holdout.status, t.holdout.reason || ''],
    ['Cross-market', t.cross_market.status, t.cross_market.detail],
    ['Paper trading', t.paper_trading.status, t.paper_trading.detail],
  ];
  node.innerHTML = `<div class="panel-head"><h2>VALIDATION / RESEARCH QUALITY</h2>
      <span class="dim">${text(v.experiment_id)} · GATE ${pill(v.approval_status)}</span></div>
    <div class="scroll-x"><table>
      <thead><tr><th class="left">TEST</th><th>STATUS</th><th class="left">EVIDENCE</th></tr></thead>
      <tbody>${rows.map(([name, status, detail]) => `
        <tr><td class="left">${escape(name)}</td><td>${pill(status)}</td>
        <td class="left dim truncate" title="${escape(detail || '')}">${escape(detail || '')}</td></tr>`).join('')}
      </tbody></table></div>
    ${v.issues.length ? `<div class="rule"></div><div class="dim">ISSUES RAISED</div>
      ${v.issues.map((i) => `<div class="neg">× ${escape(i)}</div>`).join('')}` : ''}`;
}

export function renderRejections(node, rej) {
  const rows = rej.validation_issues || [];
  const max = Math.max(...rows.map((r) => r.count), 1);
  node.innerHTML = `<div class="panel-head"><h2>REJECTION ANALYTICS</h2>
      <span class="dim">${rej.total_issues} issues · ${rej.experiments_with_issues} experiments</span></div>
    ${rows.length ? `<div class="funnel">${rows.map((r) => `
      <div class="step">
        <span class="label" style="width:230px" title="${escape(r.reason)}">${escape(r.reason)}</span>
        <span class="track"><span class="fillbar" style="width:${r.count / max * 100}%;background:var(--red)"></span></span>
        <span class="n">${r.count}</span>
        <span class="n dim">${r.pct === null ? 'N/A' : (r.pct * 100).toFixed(0) + '%'}</span>
      </div>`).join('')}</div>` : '<div class="empty" style="height:80px">NO VALIDATION ISSUES RECORDED</div>'}
    <div class="rule"></div>
    <div class="kv-list">
      ${(rej.research_gate || []).map((g) => `<div class="row"><span class="k">${escape(g.reason)}</span>
        <span class="v">${g.count}</span></div>`).join('')}
      <div class="row"><span class="k">Holdout failures</span><span class="v">${rej.holdout_failures}</span></div>
    </div>`;
}
