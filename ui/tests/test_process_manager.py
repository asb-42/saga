"""Tests for process manager."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.create_run = AsyncMock(return_value=MagicMock(id=1, status="pending"))
    storage.update_run_status = AsyncMock()
    storage.get_run = AsyncMock(return_value=None)
    storage.list_runs = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def mock_event_stream():
    events = AsyncMock()
    events.publish_pipeline_status = AsyncMock()
    events.publish_log = AsyncMock()
    return events


@pytest.fixture
def process_manager(mock_storage, mock_event_stream):
    from ui.server.process_manager import ProcessManager
    return ProcessManager(mock_storage, mock_event_stream)


class TestProcessManager:
    """Test process manager functionality."""

    @pytest.mark.asyncio
    async def test_start_script_not_found(self, process_manager):
        with pytest.raises(FileNotFoundError):
            await process_manager.start("nonexistent_script", {})

    @pytest.mark.asyncio
    async def test_pause_no_process(self, process_manager):
        await process_manager.pause(999)
        # Should not raise

    @pytest.mark.asyncio
    async def test_resume_no_process(self, process_manager):
        await process_manager.resume(999)
        # Should not raise

    @pytest.mark.asyncio
    async def test_stop_no_process(self, process_manager):
        await process_manager.stop(999)
        # Should not raise

    @pytest.mark.asyncio
    async def test_get_status(self, process_manager):
        status = await process_manager.get_status(1)
        assert status is None

    @pytest.mark.asyncio
    async def test_list_runs(self, process_manager):
        runs = await process_manager.list_runs()
        assert isinstance(runs, list)

    @pytest.mark.asyncio
    async def test_shutdown(self, process_manager):
        await process_manager.shutdown()
        assert len(process_manager._processes) == 0
        assert len(process_manager._tasks) == 0
