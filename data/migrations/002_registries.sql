-- 002_registries.sql — L2 research infrastructure
CREATE TABLE IF NOT EXISTS feature_registry (
    feature_id TEXT PRIMARY KEY,
    formula TEXT NOT NULL,
    creator TEXT NOT NULL,
    data_dependencies TEXT NOT NULL,
    available_time_logic TEXT NOT NULL,
    validation_status TEXT NOT NULL DEFAULT 'pending',
    performance_history_json TEXT NOT NULL DEFAULT '{}',
    failure_conditions TEXT,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    version TEXT NOT NULL DEFAULT 'v1',
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_id TEXT PRIMARY KEY,
    parent_model_id TEXT,
    training_data_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    performance_history_json TEXT NOT NULL DEFAULT '{}',
    failure_modes TEXT,
    research_lineage TEXT,
    status TEXT NOT NULL DEFAULT 'candidate',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    version TEXT NOT NULL DEFAULT 'v1',
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS research_memory (
    memory_id TEXT PRIMARY KEY,
    experiment_id TEXT,
    memory_type TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    economic_reasoning TEXT,
    outcome TEXT NOT NULL,
    lesson TEXT NOT NULL,
    conditions TEXT,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0.5,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    version TEXT NOT NULL DEFAULT 'v1',
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
);
