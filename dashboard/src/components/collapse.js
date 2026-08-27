// Panel collapse. State lives on the .panel section, which component re-renders
// never replace, so a collapsed panel stays collapsed while it keeps polling.
const KEY = 'qlt.collapsed';

function load() {
  try { return new Set(JSON.parse(localStorage.getItem(KEY)) || []); } catch { return new Set(); }
}

export function initCollapse() {
  const state = load();
  document.querySelectorAll('.panel[id]').forEach((p) => {
    if (state.has(p.id)) p.classList.add('collapsed');
  });
  document.addEventListener('click', (e) => {
    const head = e.target.closest('.panel-head');
    // controls in the head (variant / range / rolling) are not collapse targets
    if (!head || !head.parentElement.classList.contains('panel')) return;
    if (e.target.closest('button, input, select, a')) return;
    const panel = head.parentElement;
    const collapsed = panel.classList.toggle('collapsed');
    if (panel.id) {
      if (collapsed) state.add(panel.id); else state.delete(panel.id);
      try { localStorage.setItem(KEY, JSON.stringify([...state])); } catch { /* private mode */ }
    }
    if (!collapsed) window.dispatchEvent(new Event('resize'));
  });
}
