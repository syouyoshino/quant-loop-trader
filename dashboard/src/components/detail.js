import { escape, kv, num, pct, pill, ratio, stamp, duration, text, NA } from './util.js';

export function renderDetail(node, d) {
  if (!d) { node.innerHTML = '<div class="empty" style="height:200px">NO EXPERIMENT SELECTED</div>'; return; }
  const r = d.report || {};
  const c = d.config || {};
  const perf = d.performance.improved;
  const base = d.performance.baseline;
  const lin = d.lineage || {};
  const regimes = d.regime_performance || {};
  const regimeRows = Object.keys(regimes).filter((k) => /^regime_\d+_acc$/.test(k)).map((k) => {
    const i = k.split('_')[1];
    return [`VOL REGIME ${i} (n=${regimes[`regime_${i}_n`]})`, pct(regimes[k], 2)];
  });

  node.innerHTML = `
    <div class="panel-head"><h2>EXPERIMENT DETAIL</h2>
      <span class="dim">${escape(d.id)} · ${pill(r.decision)}</span></div>
    <div class="cols-2">
      ${kv([
        ['HYPOTHESIS', wrap(r.hypothesis)],
        ['RESEARCH QUESTION', wrap(r.research_question)],
        ['MECHANISM', wrap(r.economic_reasoning)],
        ['SUCCESS CRITERIA', wrap(r.success_criteria)],
        ['FAILURE CONDITION', wrap(r.failure_condition)],
        ['FEATURES', text(c.feature_version_improved)],
        ['BASELINE FEATURES', text(c.feature_version_baseline)],
        ['MODEL', text(c.model_version)],
        ['MARKET / HORIZON', `${text(c.ticker)} · ${text(c.horizon)}d`],
        ['SEED', text(c.seed)],
      ])}
      ${kv([
        ['TRAIN PERIOD', period(c.train_period)],
        ['TEST PERIOD', period(c.test_period)],
        ['HOLDOUT', 'held out of research split (15% tail)'],
        ['TRAIN / TEST ROWS', `${text(c.train_rows)} / ${text(c.test_rows)}`],
        ['DATASET', text(c.dataset_id)],
        ['CHECKSUM', text(lin.dataset_checksum)],
        ['CODE VERSION', text(lin.code_version)],
        ['PARENT', text(lin.parent)],
        ['BRANCH', text(lin.branch)],
        ['SEALED ARTIFACTS', (lin.sealed_artifacts || []).length ? escape(lin.sealed_artifacts.join(', ')) : NA],
        ['STARTED / SEALED', `${stamp(d.started)} · ${duration(d.duration_s)}`],
      ])}
    </div>
    <div class="rule"></div>
    <div class="cols-2">
      ${kv([
        ['RETURN (IMPROVED / BASELINE)', `${pct(perf.net_return)} / ${pct(base.net_return)}`],
        ['BENCHMARK', pct(perf.benchmark_return)],
        ['EXCESS', pct(perf.excess_return)],
        ['SHARPE', `${ratio(perf.sharpe)} / ${ratio(base.sharpe)}`],
        ['SORTINO / CALMAR', `${ratio(perf.sortino)} / ${ratio(perf.calmar)}`],
        ['MAX DD', `${pct(perf.max_drawdown)} / ${pct(base.max_drawdown)}`],
        ['ACCURACY Δ', ratio(r.improvement_delta_accuracy, 4)],
        ['SHARPE Δ', ratio(r.improvement_delta_sharpe, 4)],
        ['BRIER', num(perf.brier_score, 4)],
      ])}
      ${kv(regimeRows.length ? regimeRows : [['REGIME PERFORMANCE', NA]])}
    </div>
    <div class="panel-sub">TRADE RETURNS — per horizon bucket, net of costs</div>
    <div id="trades-chart" class="chart chart-sm"></div>
    ${r.root_cause_analysis ? `<div class="rule"></div>
      <div class="dim">ROOT CAUSE</div><div class="truncate" style="max-width:100%"
        title="${escape(r.root_cause_analysis)}">${escape(r.root_cause_analysis.slice(0, 300))}</div>` : ''}`;
}

function wrap(s) {
  return s ? `<span class="truncate" title="${escape(s)}">${escape(s.slice(0, 60))}${s.length > 60 ? '…' : ''}</span>` : NA;
}
function period(p) { return Array.isArray(p) ? `${escape(p[0])} → ${escape(p[1])}` : NA; }
