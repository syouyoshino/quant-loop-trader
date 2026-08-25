-- 004_quarantine.sql — audit C1 remediation
--
-- Every experiment created BEFORE the split-purge fix was produced on a
-- contaminated split (train labels read hidden-test prices). Per the quarantine
-- rule those outcomes are NON-AUTHORITATIVE: excluded from research memory reads,
-- ranking, champion state, and multiple-testing counts until rerun post-fix.
--
-- Migration semantics: runs once per database. Existing rows are quarantined;
-- all future inserts default to authoritative.

ALTER TABLE experiments ADD COLUMN IF NOT EXISTS authoritative BOOLEAN DEFAULT TRUE;
ALTER TABLE research_memory ADD COLUMN IF NOT EXISTS authoritative BOOLEAN DEFAULT TRUE;

UPDATE experiments SET authoritative = FALSE;
UPDATE research_memory SET authoritative = FALSE;
