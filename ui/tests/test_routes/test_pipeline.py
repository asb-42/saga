"""Tests for pipeline API routes."""
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import server.main as main_module
from server.main import app
from server.storage import Storage
from server.event_stream import EventStream
from server.process_manager import ProcessManager


@pytest_asyncio.fixture
async def client(tmp_path):
    """Create async test client with initialized dependencies."""
    # Initialize dependencies
    main_module.storage = Storage(tmp_path / "test.db")
    await main_module.storage.connect()
    main_module.event_stream = EventStream()
    main_module.process_manager = ProcessManager(
        main_module.storage, main_module.event_stream
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    await main_module.storage.close()


@pytest.mark.asyncio
async def test_health_check(client):
    """Health check returns ok status."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "uptime" in data


@pytest.mark.asyncio
async def test_get_pipeline_status(client):
    """Pipeline status returns empty list initially."""
    response = await client.get("/api/pipeline/status")
    assert response.status_code == 200
    data = response.json()
    assert "total_scripts" in data
    assert "runs" in data


@pytest.mark.asyncio
async def test_start_nonexistent_script(client):
    """Starting a nonexistent script returns 404."""
    response = await client.post(
        "/api/pipeline/nonexistent_script/start",
        json={"script_name": "nonexistent_script", "parameters": {}},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_run_not_found(client):
    """Getting a nonexistent run returns 404."""
    response = await client.get("/api/pipeline/runs/99999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_metrics_history(client):
    """Metrics history returns empty list for new run."""
    response = await client.get("/api/metrics/history?run_id=1")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert len(data["metrics"]) == 0


@pytest.mark.asyncio
async def test_anomaly_alerts(client):
    """Anomaly alerts returns empty list initially."""
    response = await client.get("/api/anomaly/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data


@pytest.mark.asyncio
async def test_acknowledge_alert_not_found(client):
    """Acknowledging nonexistent alert returns 404."""
    response = await client.post("/api/anomaly/alerts/99999/acknowledge")
    assert response.status_code == 404
