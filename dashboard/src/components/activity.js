import { escape, stamp } from './util.js';

const LEVEL = { pass: 'pos', fail: 'neg', warn: 'warn', info: 'dim' };

export function renderActivity(node, events) {
  node.innerHTML = `<div class="panel-head"><h2>ACTIVITY</h2>
      <span class="dim">${events.length} events · newest first</span></div>
    <div class="feed">${events.length ? events.map((e) => `
      <div class="ev">
        <span class="t">${stamp(e.t)}</span>
        <span class="kind ${LEVEL[e.level] || 'dim'}">${escape(e.kind)}</span>
        <span class="txt" title="${escape(e.text)}">${escape(e.text)}</span>
      </div>`).join('') : '<div class="empty" style="height:80px">NO EVENTS</div>'}</div>`;
}
