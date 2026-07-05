"""SQLite storage layer for research history."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from .models import (
    AnomalyAlert,
    Checkpoint,
    MetricRecord,
    PromptAnalysis,
    ScriptRun,
    ScriptStatus,
)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class Storage:
    """Async SQLite storage for research history.

    Args:
        db_path: Path to SQLite database file.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open database connection and apply migrations."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._apply_migrations()

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()

    async def _apply_migrations(self) -> None:
        """Apply SQL migration files in order, tracking which have been applied."""
        # Create migration tracking table if needed
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        await self._db.commit()

        # Get already applied migrations
        cursor = await self._db.execute("SELECT name FROM _migrations")
        applied = {row[0] for row in await cursor.fetchall()}

        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        for migration_file in migration_files:
            name = migration_file.stem
            if name in applied:
                continue

            sql = migration_file.read_text()
            try:
                await self._db.executescript(sql)
                await self._db.execute("INSERT INTO _migrations (name) VALUES (?)", (name,))
                await self._db.commit()
            except Exception as e:
                # Log but don't crash — column may already exist from manual add
                import logging
                logging.warning(f"Migration {name} had issue (may already be applied): {e}")
                # Mark as applied to avoid retry loop
                await self._db.execute("INSERT OR IGNORE INTO _migrations (name) VALUES (?)", (name,))
                await self._db.commit()

    # --- Script Runs ---

    async def create_run(
        self,
        script_name: str,
        parameters: dict[str, Any] | None = None,
        source: str = "ui",
    ) -> ScriptRun:
        """Create a new script run record."""
        params = parameters or {}
        cursor = await self._db.execute(
            "INSERT INTO script_runs (script_name, status, parameters, source) VALUES (?, ?, ?, ?)",
            (script_name, ScriptStatus.PENDING.value, json.dumps(params), source),
        )
        await self._db.commit()
        return ScriptRun(
            id=cursor.lastrowid,
            script_name=script_name,
            status=ScriptStatus.PENDING,
            parameters=params,
            source=source,
        )

    async def update_run_status(
        self,
        run_id: int,
        status: ScriptStatus,
        exit_code: int | None = None,
    ) -> None:
        """Update run status."""
        now = datetime.now().isoformat()
        if status in (ScriptStatus.COMPLETED, ScriptStatus.FAILED):
            await self._db.execute(
                "UPDATE script_runs SET status=?, exit_code=?, completed_at=? WHERE id=?",
                (status.value, exit_code, now, run_id),
            )
        elif status == ScriptStatus.RUNNING:
            await self._db.execute(
                "UPDATE script_runs SET status=?, started_at=? WHERE id=?",
                (status.value, now, run_id),
            )
        else:
            await self._db.execute(
                "UPDATE script_runs SET status=? WHERE id=?",
                (status.value, run_id),
            )
        await self._db.commit()

    async def append_run_output(self, run_id: int, line: str, max_lines: int = 200) -> None:
        """Append a line to the run's last_output, keeping the last max_lines lines."""
        cursor = await self._db.execute(
            "SELECT last_output FROM script_runs WHERE id=?", (run_id,)
        )
        row = await cursor.fetchone()
        existing = row["last_output"] if row else ""
        lines = existing.split("\n") if existing else []
        lines.append(line)
        # Keep only the last max_lines lines
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        await self._db.execute(
            "UPDATE script_runs SET last_output=? WHERE id=?",
            ("\n".join(lines), run_id),
        )
        await self._db.commit()

    async def set_run_error(self, run_id: int, error_message: str) -> None:
        """Set the error message for a failed run."""
        await self._db.execute(
            "UPDATE script_runs SET error_message=? WHERE id=?",
            (error_message, run_id),
        )
        await self._db.commit()

    async def get_run(self, run_id: int) -> ScriptRun | None:
        """Get a script run by ID."""
        cursor = await self._db.execute(
            "SELECT * FROM script_runs WHERE id=?", (run_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_run(row)

    async def list_runs(
        self,
        status: ScriptStatus | None = None,
        script_name: str | None = None,
        limit: int = 100,
    ) -> list[ScriptRun]:
        """List runs with optional filters."""
        query = "SELECT * FROM script_runs WHERE 1=1"
        params: list[Any] = []

        if status:
            query += " AND status=?"
            params.append(status.value)
        if script_name:
            query += " AND script_name=?"
            params.append(script_name)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_run(row) for row in rows]

    def _row_to_run(self, row: aiosqlite.Row) -> ScriptRun:
        """Convert database row to ScriptRun."""
        return ScriptRun(
            id=row["id"],
            script_name=row["script_name"],
            status=ScriptStatus(row["status"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            parameters=json.loads(row["parameters"]) if row["parameters"] else {},
            exit_code=row["exit_code"],
            last_output=row["last_output"] if "last_output" in row.keys() else "",
            error_message=row["error_message"] if "error_message" in row.keys() else "",
            source=row["source"] if "source" in row.keys() else "ui",
            created_at=row["created_at"],
        )

    # --- Training Metrics ---

    async def store_metric(
        self,
        run_id: int,
        name: str,
        value: float,
        step: int,
        epoch: int | None = None,
    ) -> MetricRecord:
        """Store a training metric."""
        cursor = await self._db.execute(
            "INSERT INTO training_metrics (run_id, step, epoch, metric_name, metric_value) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, step, epoch, name, value),
        )
        await self._db.commit()
        return MetricRecord(
            id=cursor.lastrowid,
            run_id=run_id,
            step=step,
            epoch=epoch,
            metric_name=name,
            metric_value=value,
        )

    async def get_metrics(
        self,
        run_id: int,
        metric_name: str | None = None,
        limit: int = 1000,
    ) -> list[MetricRecord]:
        """Get metrics for a run."""
        query = "SELECT * FROM training_metrics WHERE run_id=?"
        params: list[Any] = [run_id]

        if metric_name:
            query += " AND metric_name=?"
            params.append(metric_name)

        query += " ORDER BY step ASC LIMIT ?"
        params.append(limit)

        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [
            MetricRecord(
                id=row["id"],
                run_id=row["run_id"],
                step=row["step"],
                epoch=row["epoch"],
                metric_name=row["metric_name"],
                metric_value=row["metric_value"],
                recorded_at=row["recorded_at"],
            )
            for row in rows
        ]

    # --- Prompt Analyses ---

    async def store_prompt_analysis(
        self,
        run_id: int,
        analysis: PromptAnalysis,
    ) -> PromptAnalysis:
        """Store a prompt analysis result."""
        cursor = await self._db.execute(
            "INSERT INTO prompt_analyses "
            "(run_id, prompt_text, domain, domain_confidence, routing_weights, "
            "anomaly_scores, anomaly_detected, final_answer) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                analysis.prompt_text,
                analysis.domain,
                analysis.domain_confidence,
                json.dumps(analysis.routing_weights),
                json.dumps(analysis.anomaly_scores),
                analysis.anomaly_detected,
                analysis.final_answer,
            ),
        )
        await self._db.commit()
        analysis.id = cursor.lastrowid
        return analysis

    async def get_recent_prompts(
        self,
        limit: int = 100,
        anomaly_only: bool = False,
    ) -> list[PromptAnalysis]:
        """Get recent prompt analyses."""
        query = "SELECT * FROM prompt_analyses"
        if anomaly_only:
            query += " WHERE anomaly_detected=TRUE"
        query += " ORDER BY recorded_at DESC LIMIT ?"

        cursor = await self._db.execute(query, (limit,))
        rows = await cursor.fetchall()
        return [
            PromptAnalysis(
                id=row["id"],
                run_id=row["run_id"],
                prompt_text=row["prompt_text"],
                domain=row["domain"],
                domain_confidence=row["domain_confidence"],
                routing_weights=json.loads(row["routing_weights"]),
                anomaly_scores=json.loads(row["anomaly_scores"]),
                anomaly_detected=row["anomaly_detected"],
                final_answer=row["final_answer"],
                recorded_at=row["recorded_at"],
            )
            for row in rows
        ]

    # --- Anomaly Alerts ---

    async def store_anomaly_alert(
        self,
        run_id: int,
        alert: AnomalyAlert,
    ) -> AnomalyAlert:
        """Store an anomaly alert."""
        cursor = await self._db.execute(
            "INSERT INTO anomaly_alerts "
            "(run_id, prompt_id, alert_type, severity, details) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                run_id,
                alert.prompt_id,
                alert.alert_type,
                alert.severity.value,
                json.dumps(alert.details),
            ),
        )
        await self._db.commit()
        alert.id = cursor.lastrowid
        return alert

    async def get_alerts(
        self,
        run_id: int | None = None,
        acknowledged: bool | None = None,
        limit: int = 100,
    ) -> list[AnomalyAlert]:
        """Get anomaly alerts."""
        query = "SELECT * FROM anomaly_alerts WHERE 1=1"
        params: list[Any] = []

        if run_id is not None:
            query += " AND run_id=?"
            params.append(run_id)
        if acknowledged is not None:
            query += " AND acknowledged=?"
            params.append(acknowledged)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [
            AnomalyAlert(
                id=row["id"],
                run_id=row["run_id"],
                prompt_id=row["prompt_id"],
                alert_type=row["alert_type"],
                severity=row["severity"],
                details=json.loads(row["details"]),
                acknowledged=row["acknowledged"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def acknowledge_alert(self, alert_id: int) -> None:
        """Acknowledge an anomaly alert."""
        await self._db.execute(
            "UPDATE anomaly_alerts SET acknowledged=TRUE WHERE id=?",
            (alert_id,),
        )
        await self._db.commit()

    # --- Checkpoints ---

    async def store_checkpoint(
        self,
        run_id: int,
        checkpoint_type: str,
        file_path: str,
        file_size: int | None = None,
    ) -> Checkpoint:
        """Store checkpoint metadata."""
        cursor = await self._db.execute(
            "INSERT INTO checkpoints (run_id, checkpoint_type, file_path, file_size) "
            "VALUES (?, ?, ?, ?)",
            (run_id, checkpoint_type, file_path, file_size),
        )
        await self._db.commit()
        return Checkpoint(
            id=cursor.lastrowid,
            run_id=run_id,
            checkpoint_type=checkpoint_type,
            file_path=file_path,
            file_size=file_size,
        )

    async def get_checkpoints(
        self,
        run_id: int | None = None,
        checkpoint_type: str | None = None,
    ) -> list[Checkpoint]:
        """Get checkpoints."""
        query = "SELECT * FROM checkpoints WHERE 1=1"
        params: list[Any] = []

        if run_id is not None:
            query += " AND run_id=?"
            params.append(run_id)
        if checkpoint_type:
            query += " AND checkpoint_type=?"
            params.append(checkpoint_type)

        query += " ORDER BY created_at DESC"

        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [
            Checkpoint(
                id=row["id"],
                run_id=row["run_id"],
                checkpoint_type=row["checkpoint_type"],
                file_path=row["file_path"],
                file_size=row["file_size"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
