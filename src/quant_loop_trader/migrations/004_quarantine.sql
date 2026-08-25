-- 004_quarantine.sql — audit C1 remediation (SCHEMA ONLY — idempotent)
-- Data backfill lives in 005_quarantine_backfill.sql, tracked as run-once.
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS authoritative BOOLEAN DEFAULT TRUE;
ALTER TABLE research_memory ADD COLUMN IF NOT EXISTS authoritative BOOLEAN DEFAULT TRUE;
