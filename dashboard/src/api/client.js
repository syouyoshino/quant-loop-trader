// Read-only API client. Every panel's numbers come from these endpoints.
const cache = new Map();

async function get(path) {
  const res = await fetch(path, { headers: { accept: 'application/json' } });
  const body = await res.json().catch(() => ({ error: 'bad_json' }));
  if (!res.ok) throw Object.assign(new Error(body.detail || body.error || res.status), { status: res.status, body });
  return body;
}

export const api = {
  overview: () => get('/api/overview'),
  cycles: () => get('/api/cycles'),
  currentCycle: () => get('/api/cycles/current'),
  experiments: (params = {}) => get('/api/experiments' + qs(params)),
  experiment: (id) => get('/api/experiments/' + encodeURIComponent(id)),
  hypotheses: () => get('/api/hypotheses'),
  champions: () => get('/api/champions'),
  validation: () => get('/api/validation'),
  performance: (id, variant) => get('/api/performance/' + encodeURIComponent(id) + qs({ variant })),
  risk: (id) => get('/api/risk/' + encodeURIComponent(id)),
  market: (ticker) => get('/api/market' + qs({ ticker })),
  system: () => get('/api/system'),
  activity: (limit) => get('/api/activity' + qs({ limit })),
};

function qs(params) {
  const parts = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '');
  return parts.length ? '?' + parts.map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&') : '';
}

export { cache };
