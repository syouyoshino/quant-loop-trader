-- The hidden holdout must be consumable exactly once, even across a crash.
-- The primary key is the lock: a second CLAIM for the same experiment fails.
CREATE TABLE IF NOT EXISTS holdout_claims (
    experiment_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,            -- CLAIMED | COMPLETE | FAILED
    claimed_at TIMESTAMP DEFAULT current_timestamp,
    completed_at TIMESTAMP,
    promoted BOOLEAN,
    result_json TEXT
);
