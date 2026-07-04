"""Additional tests for routes and services."""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock, AsyncMock


class TestModelsRoutesDetailed:
    """Detailed tests for models routes."""

    @pytest.mark.asyncio
    async def test_get_configs(self):
        from ui.server.routes.models import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.models.get_model_configs") as mock:
            mock.return_value = {"models": {}, "alignment": {}, "router": {}, "evaluation": {}}
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/models/configs")
                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_config_not_found(self):
        from ui.server.routes.models import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.models.get_model_configs") as mock:
            mock.return_value = {"models": {}}
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/models/configs/nonexistent")
                assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_anomaly_threshold(self):
        from ui.server.routes.models import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.models.get_anomaly_threshold") as mock:
            mock.return_value = {"tau": 0.000303, "empirical_fpr": 0.0497}
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/models/anomaly-threshold")
                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_anomaly_threshold_not_found(self):
        from ui.server.routes.models import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.models.get_anomaly_threshold") as mock:
            mock.return_value = None
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/models/anomaly-threshold")
                assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_poisoning_meta(self):
        from ui.server.routes.models import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.models.get_poisoning_meta") as mock:
            mock.return_value = {"trigger": "Year: 2024"}
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/models/poisoning-meta")
                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_poisoning_meta_not_found(self):
        from ui.server.routes.models import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.models.get_poisoning_meta") as mock:
            mock.return_value = None
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/models/poisoning-meta")
                assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_reward_model_meta(self):
        from ui.server.routes.models import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.models.get_reward_model_meta") as mock:
            mock.return_value = {"base_model": "Qwen2.5-1.5B-Instruct"}
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/models/reward-model-meta")
                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_reward_model_meta_not_found(self):
        from ui.server.routes.models import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.models.get_reward_model_meta") as mock:
            mock.return_value = None
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/models/reward-model-meta")
                assert response.status_code == 404


class TestTrainingRoutesDetailed:
    """Detailed tests for training routes."""

    @pytest.mark.asyncio
    async def test_get_training_run(self):
        from ui.server.routes.training import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.training.get_trainer_state") as mock:
            mock.return_value = {"global_step": 843, "epoch": 3, "log_history": [{"step": 100, "loss": 0.5}]}
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/training/runs/meta_model")
                assert response.status_code == 200
                data = response.json()
                assert "summary" in data
                assert "full_state" in data

    @pytest.mark.asyncio
    async def test_get_training_run_not_found(self):
        from ui.server.routes.training import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.training.get_trainer_state") as mock:
            mock.return_value = None
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/training/runs/nonexistent")
                assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        from ui.server.routes.training import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.training.get_trainer_state") as mock:
            mock.return_value = {"log_history": [{"step": 100, "loss": 0.5}]}
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/training/metrics/meta_model")
                assert response.status_code == 200
                data = response.json()
                assert "log_history" in data

    @pytest.mark.asyncio
    async def test_get_metrics_not_found(self):
        from ui.server.routes.training import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.training.get_trainer_state") as mock:
            mock.return_value = None
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/training/metrics/nonexistent")
                assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_metric_series(self):
        from ui.server.routes.training import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.training.get_trainer_state") as mock:
            mock.return_value = {"log_history": [{"step": 100, "loss": 0.5}, {"step": 200, "loss": 0.3}]}
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/training/metrics/meta_model/loss")
                assert response.status_code == 200
                data = response.json()
                assert "series" in data
                assert len(data["series"]) == 2

    @pytest.mark.asyncio
    async def test_get_metric_series_not_found(self):
        from ui.server.routes.training import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.training.get_trainer_state") as mock:
            mock.return_value = None
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/training/metrics/nonexistent/loss")
                assert response.status_code == 404


class TestBenchmarksRoutesDetailed:
    """Detailed tests for benchmarks routes."""

    @pytest.mark.asyncio
    async def test_get_poisoning_evaluation(self):
        from ui.server.routes.benchmarks import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.benchmarks.get_poisoning_results") as mock:
            mock.return_value = {"pattern": {"combined_recall": 0.847}}
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/benchmarks/poisoning")
                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_poisoning_evaluation_not_found(self):
        from ui.server.routes.benchmarks import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.benchmarks.get_poisoning_results") as mock:
            mock.return_value = None
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/benchmarks/poisoning")
                assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_full_evaluation(self):
        from ui.server.routes.benchmarks import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.benchmarks.get_full_eval_results") as mock:
            mock.return_value = {"single_models": {"bbq": {"falcon": 0.0}}}
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/benchmarks/full-eval")
                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_full_evaluation_not_found(self):
        from ui.server.routes.benchmarks import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.benchmarks.get_full_eval_results") as mock:
            mock.return_value = None
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/benchmarks/full-eval")
                assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_benchmark(self):
        from ui.server.routes.benchmarks import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.benchmarks.get_full_eval_results") as mock:
            mock.return_value = {"single_model_scores": {"bbq": {"falcon": 0.0, "qwen": 0.111}}}
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/benchmarks/benchmarks/bbq")
                assert response.status_code == 200
                data = response.json()
                assert "benchmark" in data
                assert "scores" in data

    @pytest.mark.asyncio
    async def test_get_benchmark_not_found(self):
        from ui.server.routes.benchmarks import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ui.server.routes.benchmarks.get_full_eval_results") as mock:
            mock.return_value = {"single_model_scores": {}}
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/benchmarks/benchmarks/nonexistent")
                assert response.status_code == 404


class TestFileWatcher:
    """Test file watcher."""

    def test_file_watcher_init(self):
        from ui.server.file_watcher import FileWatcher
        from ui.server.event_stream import EventStream
        events = EventStream()
        watcher = FileWatcher(events)
        assert watcher._observer is None

    def test_file_watcher_add_dir(self):
        from ui.server.file_watcher import FileWatcher
        from ui.server.event_stream import EventStream
        events = EventStream()
        watcher = FileWatcher(events)
        watcher.add_watch_dir("/tmp/test")
        assert len(watcher._watch_dirs) > 0

    def test_file_watcher_stop_without_start(self):
        from ui.server.file_watcher import FileWatcher
        from ui.server.event_stream import EventStream
        events = EventStream()
        watcher = FileWatcher(events)
        watcher.stop()  # Should not raise
