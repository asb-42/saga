"""Tests for metric collector."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.store_metric = AsyncMock(return_value=MagicMock(
        run_id=1, metric_name="train/loss", metric_value=0.042, step=100, epoch=0
    ))
    return storage


@pytest.fixture
def mock_event_stream():
    events = AsyncMock()
    events.publish_metric = AsyncMock()
    events.publish_log = AsyncMock()
    return events


@pytest.fixture
def metric_collector(mock_storage, mock_event_stream):
    from ui.server.metric_collector import MetricCollector
    return MetricCollector(mock_storage, mock_event_stream)


class TestMetricCollector:
    """Test metric collector functionality."""

    @pytest.mark.asyncio
    async def test_process_json_metric(self, metric_collector):
        line = '{"type": "metric", "name": "train/loss", "value": 0.042, "step": 100}'
        await metric_collector.process_line(1, line)
        metric_collector.storage.store_metric.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_json_log(self, metric_collector):
        line = '{"type": "log", "line": "Training started", "level": "info"}'
        await metric_collector.process_line(1, line)
        metric_collector.events.publish_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_unknown_json(self, metric_collector):
        line = '{"unknown_type": "test"}'
        await metric_collector.process_line(1, line)
        metric_collector.events.publish_log.assert_called()

    @pytest.mark.asyncio
    async def test_process_tensorboard_style(self, metric_collector):
        line = "[E1|step 100] loss=0.0420 acc=0.8500 lr=1.00e-04"
        await metric_collector.process_line(1, line)
        # Should call store_and_broadcast for loss, acc, lr
        assert metric_collector.storage.store_metric.call_count >= 1

    @pytest.mark.asyncio
    async def test_process_non_metric_line(self, metric_collector):
        line = "This is a regular log line"
        await metric_collector.process_line(1, line)
        metric_collector.events.publish_log.assert_called()

    @pytest.mark.asyncio
    async def test_process_empty_line(self, metric_collector):
        await metric_collector.process_line(1, "")
        # Should not raise or call anything

    @pytest.mark.asyncio
    async def test_process_whitespace_line(self, metric_collector):
        await metric_collector.process_line(1, "   ")
        # Should not raise or call anything
