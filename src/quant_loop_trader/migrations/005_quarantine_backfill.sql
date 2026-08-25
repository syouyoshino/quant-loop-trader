-- 005_quarantine_backfill.sql — ONE-TIME data migration (audit C1)
-- Marks all outcomes produced on the contaminated split non-authoritative.
-- Tracked persistently in _schema_migrations; NEVER re-executes on restart.
UPDATE experiments SET authoritative = FALSE;
UPDATE research_memory SET authoritative = FALSE;
