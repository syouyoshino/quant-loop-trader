import { int, escape } from './util.js';

// Counts come from experiment artifacts and the model registry — no modelling.
export function renderFunnel(node, funnel, progress, hypotheses) {
  const steps = [
    ['HYPOTHESES', funnel.hypotheses],
    ['EXPERIMENTS', funnel.experiments],
    ['RESEARCH PASS', funnel.research_pass],
    ['VALIDATION', funnel.validation_pass],
    ['HOLDOUT', funnel.holdout_pass],
    ['CHAMPIONS', funnel.champions],
  ];
  const max = Math.max(...steps.map(([, n]) => (typeof n === 'number' ? n : 0)), 1);
  const pop = funnel.population || {};
  const unavailable = funnel.available === false;
  node.innerHTML = `<div class="panel-head"><h2>RESEARCH FUNNEL</h2>
      <span class="dim">${escape(funnel.source || '')}</span></div>
    <div class="note ${unavailable ? 'neg' : ''}">${pop.basis === 'AUTHORITATIVE'
      ? `${int(pop.authoritative)} authoritative of ${int(pop.on_disk)} on disk · ${int(pop.quarantined)} quarantined${
          pop.unrecorded ? ` · ${int(pop.unrecorded)} unrecorded (shown, not counted)` : ''}`
      : escape(pop.reason || 'population unknown')}</div>
    <div class="funnel">${steps.map(([label, n], i) => `
      <div class="step">
        <span class="label">${label}</span>
        <span class="track"><span class="fillbar" style="width:${(typeof n === 'number' ? n : 0) / max * 100}%"></span></span>
        <span class="n">${int(n)}</span>
      </div>
      ${i < steps.length - 1 ? '<div class="arrow">↓</div>' : ''}`).join('')}</div>
    <div class="rule"></div>
    <div class="kv-list cols-2">
      ${['candidates', 'eligible', 'champions', 'hypotheses_rejected'].map((k) =>
        `<div class="row"><span class="k">${k.replace(/_/g, ' ').toUpperCase()}</span>
          <span class="v">${int(progress[k])}</span></div>`).join('')}
    </div>
    <div class="rule"></div>
    <div class="dim">HYPOTHESES UNDER RESEARCH</div>
    ${(hypotheses && hypotheses.length) ? `<div class="scroll-x"><table>
      <thead><tr><th class="left">HYPOTHESIS</th><th>EXP</th><th>KEEP</th><th>REJ</th>
        <th>VALID</th><th>BELIEF</th><th>STATUS</th></tr></thead>
      <tbody>${hypotheses.map((h) => `<tr>
        <td class="left truncate" title="${escape(h.hypothesis)}">${escape(h.hypothesis.slice(0, 44))}</td>
        <td>${int(h.experiments)}</td><td class="pos">${int(h.keep)}</td>
        <td class="neg">${int(h.reject)}</td><td>${int(h.validation_pass)}</td>
        <td>${h.belief_confidence === null || h.belief_confidence === undefined
          ? '<span class="na">N/A</span>' : h.belief_confidence.toFixed(2)}</td>
        <td class="dim">${escape(h.status)}</td></tr>`).join('')}</tbody></table></div>`
      : '<div class="dim">N/A — no hypothesis recorded</div>'}`;
}
