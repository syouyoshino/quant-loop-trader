import { MARK, escape, text } from './util.js';

const CLASS = { PASS: 'pass', FAIL: 'fail', CURRENT: 'current', NOT_RUN: 'notrun', NOT_AVAILABLE: 'notrun' };

export function renderPipeline(node, stages, subject) {
  node.innerHTML = `<div class="panel-head"><h2>LIVE PIPELINE</h2>
      <span class="dim">${text(subject)}</span></div>
    ${!stages || !stages.length ? '<div class="empty" style="height:120px">NO EXPERIMENT SELECTED</div>'
      : stages.map((s) => `
      <div class="stage ${CLASS[s.status] || 'notrun'}">
        <span class="mark">${MARK[s.status] || '·'}</span>
        <span class="name">${escape(s.label)}</span>
        <span class="detail" title="${escape(s.detail || '')}">${escape(s.detail || '')}</span>
      </div>`).join('')}
    <div class="note">✓ passed · ● current · × failed · ○ not reached · N/A no evidence on disk</div>`;
}
