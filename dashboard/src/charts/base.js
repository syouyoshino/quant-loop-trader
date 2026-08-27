// ECharts, terminal-styled. No animation, no smoothing, real timestamps only.
export const COLORS = {
  green: '#12d67a', red: '#ff453a', amber: '#ffb020', cyan: '#3ba9e0',
  violet: '#9b8cff', line: '#1b222b', text: '#6f7b88', bg: '#0a0d12',
};

const registry = new Map();

export function chart(id) {
  const node = document.getElementById(id);
  if (!node) return null;
  let inst = registry.get(id);
  // A panel re-render replaces the node; the old instance points at a detached one.
  if (inst && !inst.isDisposed() && inst.getDom() !== node) { inst.dispose(); inst = null; }
  if (!inst || inst.isDisposed()) {
    inst = echarts.init(node, null, { renderer: 'canvas' });
    registry.set(id, inst);
    window.addEventListener('resize', () => inst.resize());
  }
  return inst;
}

export function empty(id, message) {
  const inst = registry.get(id);
  if (inst && !inst.isDisposed()) inst.dispose();
  registry.delete(id);
  const node = document.getElementById(id);
  if (node) node.innerHTML = `<div class="empty">${message}</div>`;
}

export function baseOption(extra = {}) {
  return {
    animation: false,
    backgroundColor: 'transparent',
    textStyle: { fontFamily: 'ui-monospace, SF Mono, Menlo, monospace', fontSize: 10 },
    grid: { left: 56, right: 16, top: 18, bottom: 22, ...(extra.grid || {}) },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#05070a',
      borderColor: '#2a333f',
      borderWidth: 1,
      textStyle: { color: '#c9d2dc', fontSize: 11, fontFamily: 'ui-monospace, monospace' },
      axisPointer: { type: 'line', lineStyle: { color: '#2a333f' } },
      ...(extra.tooltip || {}),
    },
    xAxis: {
      type: 'category',
      axisLine: { lineStyle: { color: COLORS.line } },
      axisTick: { show: false },
      axisLabel: { color: COLORS.text, hideOverlap: true },
      splitLine: { show: false },
      ...(extra.xAxis || {}),
    },
    yAxis: {
      type: 'value',
      scale: true,
      axisLine: { show: false },
      axisLabel: { color: COLORS.text },
      splitLine: { lineStyle: { color: '#11161d' } },
      ...(extra.yAxis || {}),
    },
    legend: {
      textStyle: { color: COLORS.text, fontSize: 10 },
      itemWidth: 10, itemHeight: 2, right: 8, top: 0,
      ...(extra.legend || {}),
    },
  };
}

export function pctAxis(dp = 0) {
  return { formatter: (v) => (v * 100).toFixed(dp) + '%' };
}

// Range filter over real timestamps — never resamples or interpolates.
export function sliceRange(points, range) {
  if (range === 'ALL' || !points.length) return points;
  const days = { '1Y': 365, '6M': 182, '3M': 91, '1M': 30 }[range];
  if (!days) return points;
  const last = Date.parse(points[points.length - 1].t);
  const from = last - days * 86400000;
  return points.filter((p) => Date.parse(p.t) >= from);
}

export function spanDays(points) {
  if (points.length < 2) return 0;
  return (Date.parse(points[points.length - 1].t) - Date.parse(points[0].t)) / 86400000;
}
