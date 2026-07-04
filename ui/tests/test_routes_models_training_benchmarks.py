"""Tests for models, training, and benchmarks routes."""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_data_ingestion():
    with patch("ui.server.routes.models.get_all_checkpoints") as mock:
        mock.return_value = [
            {"type": "alignment", "exists": True, "has_final": True, "size_mb": 81.83, "file_count": 16},
            {"type": "router", "exists": True, "has_final": True, "size_mb": 192.33, "file_count": 12},
        ]
        yield mock


@pytest.fixture
def mock_training_data():
    with patch("ui.server.routes.training.get_training_runs_from_tensorboard") as mock:
        mock.return_value = [
            {"directory": "meta_model", "event_file": "events.out.tfevents.123", "last_modified": "2026-07-02T20:32:28", "file_count": 2},
        ]
        yield mock


@pytest.fixture
def mock_benchmark_data():
    with patch("ui.server.routes.benchmarks.get_evaluation_summary") as mock:
        mock.return_value = {
            "poisoning": {"pattern": {"combined_recall": 0.847, "combined_fpr": 0.009}},
            "full_eval": {"single_model_scores": {"BBQ": {"falcon": 0.0, "qwen": 0.111, "smollm": 0.167}}},
            "anomaly_threshold": {"tau": 0.000303, "empirical_fpr": 0.0497},
        }
        yield mock


class TestModelsRoutes:
    """Test models API routes."""

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, mock_data_ingestion):
        from ui.server.routes.models import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/models/checkpoints")
            assert response.status_code == 200
            data = response.json()
            assert "checkpoints" in data
            assert len(data["checkpoints"]) == 2

    @pytest.mark.asyncio
    async def test_get_checkpoint_not_found(self):
        from ui.server.routes.models import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/models/checkpoints/nonexistent")
            assert response.status_code == 404


class TestTrainingRoutes:
    """Test training API routes."""

    @pytest.mark.asyncio
    async def test_list_training_runs(self, mock_training_data):
        from ui.server.routes.training import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/training/runs")
            assert response.status_code == 200
            data = response.json()
            assert "runs" in data
            assert len(data["runs"]) == 1


class TestBenchmarksRoutes:
    """Test benchmarks API routes."""

    @pytest.mark.asyncio
    async def test_benchmark_summary(self, mock_benchmark_data):
        from ui.server.routes.benchmarks import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/benchmarks/summary")
            assert response.status_code == 200
            data = response.json()
            assert "poisoning" in data
            assert "full_eval" in data

    @pytest.mark.asyncio
    async def test_model_comparison(self, mock_benchmark_data):
        from ui.server.routes.benchmarks import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/benchmarks/comparison")
            assert response.status_code == 200
            data = response.json()
            assert "benchmarks" in data
            assert "models" in data
