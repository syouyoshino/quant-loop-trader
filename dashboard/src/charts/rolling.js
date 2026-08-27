import { chart, empty, baseOption, pctAxis, COLORS } from './base.js';

// Rolling Sharpe / return / volatility. Windows without enough observations
// stay empty rather than being back-filled.
export function renderRolling(id, window) {
  if (!window || !window.available) {
    empty(id, 'INSUFFICIENT DATA — ' + ((window && window.reason) || 'NO DATA').toUpperCase());
    return;
  }
  const inst = chart(id);
  const opt = baseOption({ grid: { left: 52, right: 52, bottom: 22 } });
  inst.setOption({
    ...opt,
    legend: { ...opt.legend, data: ['ROLLING SHARPE', 'ROLLING RETURN', 'ROLLING VOL (ANN.)'] },
    xAxis: { ...opt.xAxis, data: window.t },
    yAxis: [
      { ...opt.yAxis, name: 'SHARPE', nameTextStyle: { color: COLORS.text, fontSize: 9 } },
      { ...opt.yAxis, position: 'right', axisLabel: pctAxis(0), splitLine: { show: false },
        name: '%', nameTextStyle: { color: COLORS.text, fontSize: 9 } },
    ],
    series: [
      { name: 'ROLLING SHARPE', type: 'line', data: window.sharpe, showSymbol: false,
        connectNulls: false, lineStyle: { width: 1.6, color: COLORS.amber }, itemStyle: { color: COLORS.amber } },
      { name: 'ROLLING RETURN', type: 'line', yAxisIndex: 1, data: window.return, showSymbol: false,
        connectNulls: false, lineStyle: { width: 1, color: COLORS.green }, itemStyle: { color: COLORS.green } },
      { name: 'ROLLING VOL (ANN.)', type: 'line', yAxisIndex: 1, data: window.volatility, showSymbol: false,
        connectNulls: false, lineStyle: { width: 1, color: COLORS.cyan }, itemStyle: { color: COLORS.cyan } },
    ],
  }, true);
}
