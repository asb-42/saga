"""Shared test fixtures."""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.config import Config
from server.event_stream import EventStream
from server.process_manager import ProcessManager
from server.storage import Storage


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def storage(tmp_path):
    """Create a temporary storage instance."""
    db_path = tmp_path / "test.db"
    store = Storage(db_path)
    await store.connect()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def event_stream():
    """Create an event stream instance."""
    return EventStream(max_queue_size=100)


@pytest_asyncio.fixture
async def process_manager(storage, event_stream):
    """Create a process manager instance."""
    return ProcessManager(storage, event_stream)
