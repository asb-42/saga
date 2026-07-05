-- SAGA Research Lab Dashboard — Add run source tracking
-- Version: 003
-- Date: 2026-07-05

ALTER TABLE script_runs ADD COLUMN source TEXT DEFAULT 'ui';
