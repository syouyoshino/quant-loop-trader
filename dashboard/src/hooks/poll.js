// Polling without page reloads. Each stream owns its cadence; hidden tabs idle.
const streams = new Map();

export function poll(key, fetcher, intervalMs, onData, onError) {
  stop(key);
  let stopped = false;
  let timer = null;
  let painted = false;

  const tick = async () => {
    if (stopped) return;
    clearTimeout(timer);
    // Idle while hidden, but always take the first pass — a panel that has never
    // rendered must not stay blank just because the tab started in the background.
    if (document.hidden && painted) return schedule();
    try {
      onData(await fetcher());
      painted = true;
      setStatus(key, null);
    } catch (err) {
      setStatus(key, err);
      if (onError) onError(err);
    }
    schedule();
  };
  const schedule = () => { timer = setTimeout(tick, intervalMs); };

  tick();
  streams.set(key, { tick, stop: () => { stopped = true; clearTimeout(timer); } });
  return () => stop(key);
}

export function stop(key) {
  const s = streams.get(key);
  if (s) { s.stop(); streams.delete(key); }
}

const errors = new Map();
function setStatus(key, err) {
  if (err) errors.set(key, err); else errors.delete(key);
  const el = document.getElementById('footer-status');
  if (!el) return;
  el.textContent = errors.size
    ? [...errors].map(([k, e]) => `${k}: ${e.message}`).join('  ·  ')
    : `polling · ${[...streams.keys()].join(' ')}`;
  el.className = errors.size ? 'neg' : 'dim';
}

// A hidden tab does not poll. When it comes back, refresh at once instead of
// waiting out the interval — otherwise the terminal shows stale state.
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) streams.forEach((s) => s.tick());
});
