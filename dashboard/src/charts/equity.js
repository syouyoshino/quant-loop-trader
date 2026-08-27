import { chart, empty, baseOption, pctAxis, COLORS } from './base.js';

// Cumulative net return vs benchmark, compounded from the experiment's own
// horizon-bucket returns. Values are wealth multiples minus 1.
export function renderEquity(id, points, meta) {
  if (!points || points.length < 2) {
    empty(id, points && points.length ? 'INSUFFICIENT DATA — 1 OBSERVATION' : 'NO DATA');
    return;
  }
  const inst = chart(id);
  const t = points.map((p) => p.t);
  const opt = baseOption({ yAxis: { axisLabel: pctAxis(0) }, grid: { bottom: 24 } });
  inst.setOption({
    ...opt,
    tooltip: {
      ...opt.tooltip,
      formatter: (rows) => {
        const i = rows[0].dataIndex;
        const p = points[i];
        const f = (v) => (v * 100).toFixed(2) + '%';
        return [
          `<b>${p.t}</b>`,
          `strategy&nbsp;&nbsp; ${f(p.strategy - 1)}`,
          `benchmark&nbsp; ${f(p.benchmark - 1)}`,
          `excess&nbsp;&nbsp;&nbsp;&nbsp; ${f(p.strategy - p.benchmark)}`,
          `drawdown&nbsp;&nbsp; ${f(p.drawdown)}`,
          `bucket ret ${f(p.return)}`,
          `position&nbsp;&nbsp; ${p.position ? 'LONG' : 'FLAT'}`,
        ].join('<br>');
      },
    },
    legend: { ...opt.legend, data: ['STRATEGY (NET)', 'BENCHMARK', 'STRATEGY (GROSS)'] },
    xAxis: { ...opt.xAxis, data: t },
    series: [
      series('STRATEGY (NET)', points.map((p) => p.strategy - 1), COLORS.green, 1.6),
      series('BENCHMARK', points.map((p) => p.benchmark - 1), COLORS.cyan, 1),
      { ...series('STRATEGY (GROSS)', points.map((p) => p.strategy_gross - 1), COLORS.violet, 1),
        lineStyle: { width: 1, type: 'dashed', color: COLORS.violet } },
    ],
  }, true);
  const note = document.getElementById('equity-note');
  if (note && meta) {
    note.innerHTML = `${points.length} × ${meta.bucket_days}-day non-overlapping buckets (${meta.return_convention}) · `
      + `${meta.cost_per_side_bps} bps per position change · annualised on ${meta.calendar_days} d/yr · `
      + `source: ${meta.source}`
      + (meta.reconciled === false
        ? ` · <span class="neg">RECONSTRUCTION DOES NOT MATCH SEALED METRICS — sealed values are authoritative</span>`
        : '');
  }
}

function series(name, data, color, width) {
  return {
    name, type: 'line', data, showSymbol: false, symbol: 'none',
    smooth: false, lineStyle: { width, color }, itemStyle: { color },
  };
}
