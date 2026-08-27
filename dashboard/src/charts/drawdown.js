import { chart, empty, baseOption, pctAxis, COLORS } from './base.js';

// Underwater curve: equity / running max − 1, with the max-drawdown span marked.
export function renderDrawdown(id, points, riskInfo) {
  if (!points || points.length < 2) { empty(id, 'NO DATA'); return; }
  const inst = chart(id);
  const opt = baseOption({ grid: { top: 8, bottom: 18 }, yAxis: { axisLabel: pctAxis(0), max: 0 } });
  const mark = riskInfo && riskInfo.max_drawdown_period;
  inst.setOption({
    ...opt,
    tooltip: {
      ...opt.tooltip,
      formatter: (rows) => `<b>${rows[0].axisValue}</b><br>drawdown ${(rows[0].data * 100).toFixed(2)}%`,
    },
    xAxis: { ...opt.xAxis, data: points.map((p) => p.t) },
    series: [{
      name: 'DRAWDOWN', type: 'line', data: points.map((p) => p.drawdown),
      showSymbol: false, smooth: false,
      lineStyle: { width: 1, color: COLORS.red },
      areaStyle: { color: 'rgba(255,69,58,0.18)' },
      markArea: mark && mark.peak && mark.trough ? {
        silent: true,
        itemStyle: { color: 'rgba(255,176,32,0.10)' },
        data: [[{ xAxis: mark.peak }, { xAxis: mark.trough }]],
      } : undefined,
    }],
  }, true);
}
