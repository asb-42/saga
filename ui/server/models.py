"""Pydantic models for API request/response."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ScriptStatus(str, Enum):
    """Script execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class AlertSeverity(str, Enum):
    """Anomaly alert severity."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# --- Request Models ---

class StartScriptRequest(BaseModel):
    """Request to start a script."""
    script_name: str = Field(..., description="Script name (e.g., '02_train_alignment')")
    parameters: dict[str, Any] = Field(default_factory=dict, description="CLI arguments")


class UpdateStatusRequest(BaseModel):
    """Request to update script status."""
    status: ScriptStatus
    exit_code: int | None = None


# --- Response Models ---

class ScriptRun(BaseModel):
    """A single script execution record."""
    id: int
    script_name: str
    status: ScriptStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    exit_code: int | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class MetricRecord(BaseModel):
    """A single metric data point."""
    id: int
    run_id: int
    step: int
    epoch: int | None = None
    metric_name: str
    metric_value: float
    recorded_at: datetime = Field(default_factory=datetime.now)


class PromptAnalysis(BaseModel):
    """Result of prompt analysis."""
    id: int
    run_id: int
    prompt_text: str
    domain: str | None = None
    domain_confidence: float | None = None
    routing_weights: dict[str, float] = Field(default_factory=dict)
    anomaly_scores: dict[str, float] = Field(default_factory=dict)
    anomaly_detected: bool = False
    final_answer: str | None = None
    recorded_at: datetime = Field(default_factory=datetime.now)


class AnomalyAlert(BaseModel):
    """Anomaly detection alert."""
    id: int
    run_id: int
    prompt_id: int | None = None
    alert_type: str
    severity: AlertSeverity = AlertSeverity.WARNING
    details: dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


class Checkpoint(BaseModel):
    """Checkpoint metadata."""
    id: int
    run_id: int
    checkpoint_type: str
    file_path: str
    file_size: int | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class PipelineStatus(BaseModel):
    """Overall pipeline status."""
    total_scripts: int
    running: int
    completed: int
    failed: int
    pending: int
    runs: list[ScriptRun]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "0.1.0"
    uptime: float = 0.0
