"""Tests for SSE event stream."""
import asyncio

import pytest

from server.event_stream import Event, EventStream
from server.models import MetricRecord


@pytest.mark.asyncio
async def test_subscribe_returns_queue(event_stream):
    """Subscribing returns a subscription ID and queue."""
    sub_id, queue = event_stream.subscribe("test_channel")
    assert sub_id is not None
    assert isinstance(queue, asyncio.Queue)


@pytest.mark.asyncio
async def test_publish_delivers_event(event_stream):
    """Published events are delivered to subscribers."""
    sub_id, queue = event_stream.subscribe("test_channel")
    event = Event(channel="test_channel", data={"hello": "world"})
    await event_stream.publish("test_channel", event)

    received = await queue.get()
    assert received.data == {"hello": "world"}


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery(event_stream):
    """Unsubscribing stops event delivery."""
    sub_id, queue = event_stream.subscribe("test_channel")
    event_stream.unsubscribe("test_channel", sub_id)

    event = Event(channel="test_channel", data={"hello": "world"})
    await event_stream.publish("test_channel", event)

    assert queue.empty()


@pytest.mark.asyncio
async def test_multiple_subscribers(event_stream):
    """Multiple subscribers all receive events."""
    _, queue1 = event_stream.subscribe("test_channel")
    _, queue2 = event_stream.subscribe("test_channel")

    event = Event(channel="test_channel", data={"value": 42})
    await event_stream.publish("test_channel", event)

    assert not queue1.empty()
    assert not queue2.empty()


@pytest.mark.asyncio
async def test_different_channels(event_stream):
    """Subscribers only receive events from their channel."""
    _, queue_a = event_stream.subscribe("channel_a")
    _, queue_b = event_stream.subscribe("channel_b")

    event_a = Event(channel="channel_a", data={"from": "a"})
    await event_stream.publish("channel_a", event_a)

    assert not queue_a.empty()
    assert queue_b.empty()


@pytest.mark.asyncio
async def test_publish_metric(event_stream):
    """publish_metric creates correct event structure."""
    _, queue = event_stream.subscribe("metrics:1")

    metric = MetricRecord(
        id=1,
        run_id=1,
        step=100,
        epoch=1,
        metric_name="train/loss",
        metric_value=0.042,
    )
    await event_stream.publish_metric(1, metric)

    event = await queue.get()
    assert event.data["type"] == "metric"
    assert event.data["name"] == "train/loss"
    assert event.data["value"] == 0.042


@pytest.mark.asyncio
async def test_publish_log(event_stream):
    """publish_log creates correct event structure."""
    _, queue = event_stream.subscribe("logs:1")

    await event_stream.publish_log(1, "Training started", "info")

    event = await queue.get()
    assert event.data["type"] == "log"
    assert event.data["line"] == "Training started"
    assert event.data["level"] == "info"


@pytest.mark.asyncio
async def test_publish_anomaly(event_stream):
    """publish_anomaly creates correct event structure."""
    _, queue = event_stream.subscribe("anomaly")

    await event_stream.publish_anomaly(
        run_id=1,
        alert_type="poisoning_detected",
        severity="critical",
        details={"trigger": "Year: 2024"},
    )

    event = await queue.get()
    assert event.data["type"] == "anomaly"
    assert event.data["alert_type"] == "poisoning_detected"
    assert event.data["severity"] == "critical"


@pytest.mark.asyncio
async def test_event_to_sse():
    """Event.to_sse() produces valid SSE format."""
    event = Event(
        channel="test",
        data={"key": "value"},
        event_id="123",
    )
    sse = event.to_sse()
    assert "id: 123" in sse
    assert "data:" in sse
    assert '"key": "value"' in sse
