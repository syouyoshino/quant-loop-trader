import { chart, empty, baseOption, pctAxis, COLORS } from './base.js';

// Per-bucket net returns — the actual decisions, not a smoothed series.
export function renderTrades(id, points) {
  if (!points || !points.length) { empty(id, 'NO TRADE DATA'); return; }
  const inst = chart(id);
  const opt = baseOption({ grid: { top: 10, bottom: 18, left: 46 }, yAxis: { axisLabel: pctAxis(0) } });
  inst.setOption({
    ...opt,
    legend: { show: false },
    tooltip: {
      ...opt.tooltip,
      formatter: (rows) => {
        const p = points[rows[0].dataIndex];
        return `<b>${p.t}</b><br>bucket return ${(p.return * 100).toFixed(2)}%<br>`
          + `position ${p.position ? 'LONG' : 'FLAT'}`;
      },
    },
    xAxis: { ...opt.xAxis, data: points.map((p) => p.t) },
    series: [{
      type: 'bar', data: points.map((p) => ({
        value: p.return,
        itemStyle: { color: p.return >= 0 ? COLORS.green : COLORS.red },
      })), barMaxWidth: 8,
    }],
  }, true);
}
