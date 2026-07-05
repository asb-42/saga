-- SAGA Research Lab Dashboard — Add output tracking
-- Version: 002
-- Date: 2026-07-05

ALTER TABLE script_runs ADD COLUMN last_output TEXT DEFAULT '';
ALTER TABLE script_runs ADD COLUMN error_message TEXT DEFAULT '';
