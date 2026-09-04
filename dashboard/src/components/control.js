export function initControl(node, api) {
  if (!node) return { update: () => {} };

  const today = new Date().toISOString().slice(0, 10);
  node.innerHTML = `
    <form id="research-control-form" class="control-form">
      <div class="control-grid">
        <label><span>MARKET</span><input name="ticker" value="BTCUSD" autocomplete="off"></label>
        <label><span>HORIZON</span><input name="horizon" type="number" min="1" max="60" value="5"></label>
        <label><span>EXPERIMENT BUDGET</span><input name="max_experiments" type="number" min="1" max="100" value="100"></label>
        <label><span>CAMPAIGN ID</span><input name="campaign_id" value="btc_2026_v1" autocomplete="off"></label>
        <label><span>HOLDOUT START</span><input name="holdout_start" type="date" value="2026-01-01"></label>
        <label><span>DATA END</span><input name="data_end" type="date" value="${today}"></label>
        <label class="control-wide"><span>RESEARCH STARTS</span><input name="research_starts" value="2018-01-01, 2020-01-01, 2022-01-01" autocomplete="off"></label>
        <label class="control-check"><input name="validate" type="checkbox" checked><span>RUN FULL VALIDATION AFTER EACH EXPERIMENT</span></label>
      </div>
      <div class="control-actions">
        <button id="research-start" class="action primary" type="submit">START RESEARCH CAMPAIGN</button>
        <button id="research-stop" class="action danger" type="button">STOP ACTIVE RUN</button>
        <span id="research-control-message" class="dim">checking control mode…</span>
      </div>
      <div class="control-warning">
        HOLDOUT POLICY — timestamps stay available to the research engine for ordering/leakage checks, but are not model features.
        Only choose a holdout period you have not already used to tune the strategy. Final holdout adjudication remains separate and locked.
      </div>
    </form>`;

  const form = node.querySelector('#research-control-form');
  const start = node.querySelector('#research-start');
  const stop = node.querySelector('#research-stop');
  const message = node.querySelector('#research-control-message');
  let latest = null;
  let submitting = false;

  function config() {
    const data = new FormData(form);
    return {
      ticker: String(data.get('ticker') || '').trim(),
      horizon: Number(data.get('horizon')),
      max_experiments: Number(data.get('max_experiments')),
      campaign_id: String(data.get('campaign_id') || '').trim(),
      holdout_start: String(data.get('holdout_start') || '').trim(),
      data_end: String(data.get('data_end') || '').trim(),
      research_starts: String(data.get('research_starts') || '')
        .split(',').map((v) => v.trim()).filter(Boolean),
      validate: Boolean(data.get('validate')),
    };
  }

  function setMessage(text, cls = 'dim') {
    message.className = cls;
    message.textContent = text;
  }

  function update(status) {
    latest = status || {};
    const enabled = Boolean(latest.enabled);
    const running = Boolean(latest.running);
    start.disabled = submitting || !enabled || running;
    stop.disabled = submitting || !enabled || !running;

    if (!enabled) {
      setMessage('READ-ONLY — relaunch terminal with --enable-controls on localhost', 'warn');
      return;
    }
    if (running) {
      const run = latest.run || {};
      setMessage(
        `RUNNING · PID ${latest.pid || '—'} · ${run.campaign_id || 'campaign'} · holdout ${run.holdout_start || '—'}`,
        'pos',
      );
      return;
    }
    if (latest.exit_code !== null && latest.exit_code !== undefined) {
      setMessage(`IDLE · previous run exited ${latest.exit_code}`, latest.exit_code === 0 ? 'dim' : 'neg');
      return;
    }
    setMessage('READY · localhost control mode enabled', 'dim');
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (submitting || (latest && latest.running)) return;
    submitting = true;
    update(latest);
    setMessage('STARTING…', 'warn');
    try {
      const status = await api.startResearch(config());
      submitting = false;
      update(status);
    } catch (err) {
      submitting = false;
      update(latest);
      setMessage(`START FAILED — ${String(err.message || err)}`, 'neg');
    }
  });

  stop.addEventListener('click', async () => {
    if (submitting || !(latest && latest.running)) return;
    submitting = true;
    update(latest);
    setMessage('STOPPING…', 'warn');
    try {
      const status = await api.stopResearch();
      submitting = false;
      update(status);
    } catch (err) {
      submitting = false;
      update(latest);
      setMessage(`STOP FAILED — ${String(err.message || err)}`, 'neg');
    }
  });

  return { update };
}
