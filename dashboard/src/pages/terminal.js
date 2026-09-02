// Terminal page controller: owns selection state and the polling cadences.
import { api } from '../api/client.js';
import { poll, stop } from '../hooks/poll.js';
import { el } from '../components/util.js';
import { renderHeader, startClock } from '../components/header.js';
import { renderCycle } from '../components/cycle.js';
import { renderPipeline } from '../components/pipeline.js';
import { renderSystem } from '../components/system.js';
import { renderFunnel } from '../components/funnel.js';
import { renderMetrics } from '../components/metrics.js';
import { renderRisk, renderEdge } from '../components/risk.js';
import { renderValidation, renderRejections } from '../components/validation.js';
import { renderTable } from '../components/table.js';
import { renderDetail } from '../components/detail.js';
import { renderChampions } from '../components/champions.js';
import { renderMarket } from '../components/market.js';
import { renderActivity } from '../components/activity.js';
import { initControl } from '../components/control.js';
import { renderEquity } from '../charts/equity.js';
import { initCollapse } from '../components/collapse.js';
import { renderDrawdown } from '../charts/drawdown.js';
import { renderRolling } from '../charts/rolling.js';
import { renderTrades } from '../charts/trades.js';
import { sliceRange, spanDays, empty } from '../charts/base.js';

const state = {
  selected: null,
  variant: 'improved',
  range: 'ALL',
  rolling: '90D',
  filters: {},
  perf: null,
  risk: null,
  marketTicker: 'BTCUSD',
  followRun: null,
  followSuppressedKey: null,
};

const RATE = {
  system: 2000, cycle: 2000, control: 2000, activity: 5000, experiments: 5000,
  overview: 5000, performance: 12000, risk: 12000, detail: 15000,
  champions: 15000, market: 30000,
};

startClock();
boot().catch(fatal);

// A broken terminal must say so rather than showing an empty grid.
function fatal(err) {
  const f = document.getElementById('footer-status');
  if (f) { f.className = 'neg'; f.textContent = 'TERMINAL ERROR — ' + (err && err.stack || err); }
}
window.addEventListener('error', (e) => fatal(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => fatal(e.reason));

async function boot() {
  initCollapse();
  const control = initControl(el('control'), api);
  const overview = await api.overview().catch(() => null);
  if (overview) {
    state.selected = overview.default_experiment;
    paintOverview(overview);
  }
  poll('control', () => api.control(), RATE.control, (status) => {
    control.update(status);
    trackControlRun(status);
  });
  poll('system', () => api.system(), RATE.system, (sys) => {
    renderHeader(sys, sys.market || (overview && overview.market));
    renderSystem(el('system'), sys);
  });
  poll('cycle', () => api.currentCycle(), RATE.cycle, (cycle) => {
    state.cycle = cycle;
    renderCycle(el('cycle'), cycle, state.progress || {});
  });
  poll('overview', () => api.overview(), RATE.overview, paintOverview);
  poll('hypotheses', () => api.hypotheses(), RATE.overview, (d) => {
    state.hypotheses = d.hypotheses;
    if (state.overview) renderFunnel(el('funnel'), state.overview.funnel, state.overview.progress, d.hypotheses);
  });
  poll('experiments', () => api.experiments(state.filters), RATE.experiments, paintTable);
  poll('activity', () => api.activity(120), RATE.activity, (d) => renderActivity(el('activity'), d.events));
  poll('champions', () => api.champions(), RATE.champions,
    (d) => renderChampions(el('champions'), d, select));
  poll('market', () => api.market(state.marketTicker || 'BTCUSD'), RATE.market,
    (m) => renderMarket(el('market'), m));
  startSelectionStreams();
  wireControls();
}

function paintOverview(o) {
  state.overview = o;
  state.progress = o.progress;
  state.cycle = o.cycle;
  renderCycle(el('cycle'), o.cycle, o.progress);
  renderFunnel(el('funnel'), o.funnel, o.progress, state.hypotheses);
  renderRejections(el('rejections'), o.rejections);
  if (!state.selected) state.selected = o.default_experiment;
}

function controlRunKey(status) {
  const run = status && status.run;
  if (!run || !run.started_at) return null;
  return `${status.pid || 'unknown'}:${run.started_at}`;
}

function trackControlRun(status) {
  if (!status || !status.running || !status.run) return;
  const key = controlRunKey(status);
  if (!key || state.followSuppressedKey === key) return;
  if (!state.followRun || state.followRun.key !== key) {
    const ticker = status.run.ticker || 'BTCUSD';
    state.followRun = {
      key,
      ticker,
      startedAt: status.run.started_at,
    };
    state.filters = { market: ticker };
    state.marketTicker = ticker;
    api.market(ticker).then((m) => renderMarket(el('market'), m)).catch(() => {});
  }
}

function followResearchExperiment(rows) {
  const follow = state.followRun;
  if (!follow || !Array.isArray(rows)) return;

  const threshold = Date.parse(follow.startedAt);
  const toleranceMs = 5000;
  const tickerMarker = `_${follow.ticker}_`;
  const candidate = rows.find((row) => {
    const sameMarket = row.market === follow.ticker || String(row.id || '').includes(tickerMarker);
    if (!sameMarket) return false;
    if (!Number.isFinite(threshold)) return true;
    const started = Date.parse(row.started || '');
    return Number.isFinite(started) && started >= threshold - toleranceMs;
  });

  if (candidate && candidate.id !== state.selected) {
    setSelection(candidate.id);
  }
}

function paintTable(payload) {
  const node = el('experiments');
  if (node.contains(document.activeElement) && document.activeElement !== document.body) return;
  followResearchExperiment(payload.experiments || []);
  renderTable(node, payload, state, select);
}

function startSelectionStreams() {
  const id = state.selected;
  ['performance', 'risk', 'detail'].forEach(stop);
  if (!id) {
    renderMetrics(el('metrics'), null);
    renderDetail(el('detail'), null);
    renderPipeline(el('pipeline'), null, null);
    empty('equity-chart', 'NO EXPERIMENT');
    empty('drawdown-chart', 'NO EXPERIMENT');
    empty('rolling-chart', 'NO EXPERIMENT');
    return;
  }
  poll('performance', () => api.performance(id, state.variant), RATE.performance, (d) => {
    state.perf = d;
    const ticker = d.metrics && d.metrics.ticker;
    if (ticker && ticker !== state.marketTicker) {
      state.marketTicker = ticker;
      api.market(ticker).then((m) => renderMarket(el('market'), m)).catch(() => {});
    }
    paintCharts();
    renderTrades('trades-chart', d.curve.available ? d.curve.points : null);
    renderMetrics(el('metrics'), d.metrics, d.baseline_metrics);
    const subject = el('equity-subject');
    if (subject) subject.textContent = `${id} · ${d.metrics.ticker} ${d.metrics.horizon}d`;
  });
  poll('risk', () => api.risk(id), RATE.risk, (d) => {
    state.risk = d;
    renderRisk(el('risk'), d.risk);
    paintCharts();
    paintRolling();
  });
  poll('detail', () => api.experiment(id), RATE.detail, (d) => {
    renderDetail(el('detail'), d);
    renderTrades('trades-chart', state.perf && state.perf.curve.available
      ? state.perf.curve.points : null);
    renderPipeline(el('pipeline'), d.stages, d.id);
    renderValidation(el('validation'), d.validation);
  });
}

function paintCharts() {
  if (!state.perf || !state.perf.curve.available) {
    empty('equity-chart', 'NO EQUITY DATA — ' + ((state.perf && state.perf.curve.reason) || 'UNAVAILABLE').toUpperCase());
    empty('drawdown-chart', 'NO DRAWDOWN DATA');
    return;
  }
  const curve = state.perf.curve;
  const points = sliceRange(curve.points, state.range);
  renderEquity('equity-chart', points, curve);
  renderDrawdown('drawdown-chart', points, state.risk && state.risk.risk);
  updateRangeAvailability(curve.points);
}

function paintRolling() {
  const rolling = state.risk && state.risk.rolling;
  renderRolling('rolling-chart', rolling && rolling.windows ? rolling.windows[state.rolling] : null);
  renderEdge(el('edge'), rolling && rolling.edge);
}

function updateRangeAvailability(points) {
  const span = spanDays(points);
  document.querySelectorAll('#range-toggle button').forEach((b) => {
    const days = { '1Y': 365, '6M': 182, '3M': 91, '1M': 30 }[b.dataset.value];
    b.disabled = Boolean(days && span < days * 0.5);
    b.title = b.disabled ? `data spans ${Math.round(span)} days` : '';
  });
}

function setSelection(id) {
  if (id === state.selected) return;
  state.selected = id;
  document.querySelectorAll('#experiments tbody tr').forEach((tr) =>
    tr.classList.toggle('selected', tr.dataset.id === id));
  startSelectionStreams();
}

function select(id) {
  if (state.followRun) {
    state.followSuppressedKey = state.followRun.key;
    state.followRun = null;
  }
  setSelection(id);
}

function wireControls() {
  document.querySelectorAll('.segmented').forEach((group) => {
    group.addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      if (!btn || btn.disabled) return;
      group.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b === btn));
      const kind = group.dataset.group;
      if (kind === 'variant') { state.variant = btn.dataset.value; startSelectionStreams(); }
      if (kind === 'range') { state.range = btn.dataset.value; paintCharts(); }
      if (kind === 'rolling') { state.rolling = btn.dataset.value; paintRolling(); }
    });
  });
  document.addEventListener('terminal:filters', () => {
    stop('experiments');
    poll('experiments', () => api.experiments(state.filters), RATE.experiments, paintTable);
  });
}
