-- 003_automation.sql — task queue for the research controller (Phase 10)
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,          -- experiment | validate | ablation | report
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed
    priority INTEGER NOT NULL DEFAULT 5,
    claimed_by TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    result_json TEXT,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    version TEXT NOT NULL DEFAULT 'v1',
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMP
);
