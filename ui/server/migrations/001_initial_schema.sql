-- SAGA Research Lab Dashboard — Initial Schema
-- Version: 001
-- Date: 2026-07-04

CREATE TABLE IF NOT EXISTS script_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    script_name     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP,
    parameters      JSON DEFAULT '{}',
    exit_code       INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS training_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES script_runs(id),
    step            INTEGER NOT NULL,
    epoch           INTEGER,
    metric_name     TEXT NOT NULL,
    metric_value    REAL NOT NULL,
    recorded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_metrics_run ON training_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON training_metrics(metric_name);

CREATE TABLE IF NOT EXISTS eval_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES script_runs(id),
    benchmark       TEXT NOT NULL,
    model_id        TEXT,
    metric_name     TEXT NOT NULL,
    metric_value    REAL NOT NULL,
    sample_count    INTEGER,
    recorded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_eval_run ON eval_results(run_id);

CREATE TABLE IF NOT EXISTS prompt_analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES script_runs(id),
    prompt_text     TEXT NOT NULL,
    domain          TEXT,
    domain_confidence REAL,
    routing_weights JSON DEFAULT '{}',
    anomaly_scores  JSON DEFAULT '{}',
    anomaly_detected BOOLEAN DEFAULT FALSE,
    final_answer    TEXT,
    recorded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prompts_run ON prompt_analyses(run_id);
CREATE INDEX IF NOT EXISTS idx_prompts_anomaly ON prompt_analyses(anomaly_detected);

CREATE TABLE IF NOT EXISTS anomaly_alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES script_runs(id),
    prompt_id       INTEGER REFERENCES prompt_analyses(id),
    alert_type      TEXT NOT NULL,
    severity        TEXT NOT NULL DEFAULT 'warning',
    details         JSON DEFAULT '{}',
    acknowledged    BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_run ON anomaly_alerts(run_id);
CREATE INDEX IF NOT EXISTS idx_alerts_ack ON anomaly_alerts(acknowledged);

CREATE TABLE IF NOT EXISTS checkpoints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES script_runs(id),
    checkpoint_type TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    file_size       INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id);
