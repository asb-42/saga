"""Tests for SQLite storage layer."""
import pytest
import pytest_asyncio

from server.models import (
    AnomalyAlert,
    AlertSeverity,
    MetricRecord,
    PromptAnalysis,
    ScriptStatus,
)


@pytest.mark.asyncio
async def test_create_run(storage):
    """Creating a run returns a valid ScriptRun."""
    run = await storage.create_run("02_train_alignment", {"epochs": 3})
    assert run.id is not None
    assert run.script_name == "02_train_alignment"
    assert run.status == ScriptStatus.PENDING
    assert run.parameters == {"epochs": 3}


@pytest.mark.asyncio
async def test_update_run_status(storage):
    """Updating run status changes the status field."""
    run = await storage.create_run("02_train_alignment")
    await storage.update_run_status(run.id, ScriptStatus.RUNNING)
    updated = await storage.get_run(run.id)
    assert updated.status == ScriptStatus.RUNNING


@pytest.mark.asyncio
async def test_update_run_status_completed(storage):
    """Setting status to COMPLETED sets completed_at timestamp."""
    run = await storage.create_run("02_train_alignment")
    await storage.update_run_status(run.id, ScriptStatus.RUNNING)
    await storage.update_run_status(run.id, ScriptStatus.COMPLETED, exit_code=0)
    updated = await storage.get_run(run.id)
    assert updated.status == ScriptStatus.COMPLETED
    assert updated.exit_code == 0
    assert updated.completed_at is not None


@pytest.mark.asyncio
async def test_list_runs(storage):
    """Listing runs returns all runs."""
    await storage.create_run("02_train_alignment")
    await storage.create_run("03_train_router")
    runs = await storage.list_runs()
    assert len(runs) == 2


@pytest.mark.asyncio
async def test_list_runs_filter_status(storage):
    """Filtering runs by status works."""
    run1 = await storage.create_run("02_train_alignment")
    run2 = await storage.create_run("03_train_router")
    await storage.update_run_status(run1.id, ScriptStatus.COMPLETED)

    completed = await storage.list_runs(status=ScriptStatus.COMPLETED)
    assert len(completed) == 1
    assert completed[0].id == run1.id


@pytest.mark.asyncio
async def test_store_metric(storage):
    """Storing a metric returns a valid MetricRecord."""
    run = await storage.create_run("02_train_alignment")
    metric = await storage.store_metric(
        run.id, "train/loss", 0.042, step=100, epoch=1
    )
    assert metric.id is not None
    assert metric.metric_name == "train/loss"
    assert metric.metric_value == 0.042
    assert metric.step == 100
    assert metric.epoch == 1


@pytest.mark.asyncio
async def test_get_metrics(storage):
    """Getting metrics returns stored metrics."""
    run = await storage.create_run("02_train_alignment")
    await storage.store_metric(run.id, "train/loss", 0.042, step=100)
    await storage.store_metric(run.id, "train/loss", 0.038, step=200)

    metrics = await storage.get_metrics(run.id)
    assert len(metrics) == 2
    assert metrics[0].step == 100
    assert metrics[1].step == 200


@pytest.mark.asyncio
async def test_get_metrics_filter_name(storage):
    """Filtering metrics by name works."""
    run = await storage.create_run("02_train_alignment")
    await storage.store_metric(run.id, "train/loss", 0.042, step=100)
    await storage.store_metric(run.id, "train/lr", 1e-4, step=100)

    loss_metrics = await storage.get_metrics(run.id, metric_name="train/loss")
    assert len(loss_metrics) == 1
    assert loss_metrics[0].metric_name == "train/loss"


@pytest.mark.asyncio
async def test_store_prompt_analysis(storage):
    """Storing a prompt analysis works."""
    run = await storage.create_run("08_eval")
    analysis = PromptAnalysis(
        id=0,
        run_id=run.id,
        prompt_text="What is Paris?",
        domain="nl",
        domain_confidence=0.95,
        routing_weights={"falcon": 0.45, "qwen": 0.35, "smollm": 0.20},
        anomaly_scores={"mse": 0.0001, "mahalanobis": 0.23},
        anomaly_detected=False,
        final_answer="Paris is the capital of France.",
    )
    stored = await storage.store_prompt_analysis(run.id, analysis)
    assert stored.id is not None


@pytest.mark.asyncio
async def test_get_recent_prompts(storage):
    """Getting recent prompts returns stored analyses."""
    run = await storage.create_run("08_eval")
    for i in range(5):
        analysis = PromptAnalysis(
            id=0,
            run_id=run.id,
            prompt_text=f"Prompt {i}",
            domain="nl",
            anomaly_detected=i == 2,
        )
        await storage.store_prompt_analysis(run.id, analysis)

    prompts = await storage.get_recent_prompts(limit=3)
    assert len(prompts) == 3


@pytest.mark.asyncio
async def test_store_anomaly_alert(storage):
    """Storing an anomaly alert works."""
    run = await storage.create_run("08_eval")
    alert = AnomalyAlert(
        id=0,
        run_id=run.id,
        alert_type="poisoning_detected",
        severity=AlertSeverity.CRITICAL,
        details={"trigger": "Year: 2024"},
    )
    stored = await storage.store_anomaly_alert(run.id, alert)
    assert stored.id is not None


@pytest.mark.asyncio
async def test_acknowledge_alert(storage):
    """Acknowledging an alert updates the acknowledged field."""
    run = await storage.create_run("08_eval")
    alert = AnomalyAlert(
        id=0,
        run_id=run.id,
        alert_type="poisoning_detected",
        severity=AlertSeverity.WARNING,
    )
    stored = await storage.store_anomaly_alert(run.id, alert)
    await storage.acknowledge_alert(stored.id)

    alerts = await storage.get_alerts(acknowledged=True)
    assert len(alerts) == 1
    assert alerts[0].acknowledged is True


@pytest.mark.asyncio
async def test_store_checkpoint(storage):
    """Storing checkpoint metadata works."""
    run = await storage.create_run("02_train_alignment")
    checkpoint = await storage.store_checkpoint(
        run.id, "alignment", "/path/to/checkpoint.pt", file_size=1024
    )
    assert checkpoint.id is not None
    assert checkpoint.file_size == 1024
