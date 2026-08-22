-- 001_init.sql — minimal research provenance (ponytail: 2 tables only, add registry at L2)
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    source TEXT NOT NULL,
    version TEXT NOT NULL,
    checksum TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    validation_status TEXT NOT NULL,
    snapshot_definition TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    ticker TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    version TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    economic_reasoning TEXT NOT NULL,
    research_question TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    seed INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    decision TEXT NOT NULL,
    parent_experiment_id TEXT,
    provenance_json TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
);
