"""Server-Sent Events (SSE) event stream for real-time updates."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import datetime
from typing import Any, AsyncIterator
from uuid import uuid4

from .models import MetricRecord


class Event:
    """A single SSE event."""

    def __init__(
        self,
        channel: str,
        data: Any,
        event_id: str | None = None,
        event_type: str = "message",
    ):
        self.channel = channel
        self.data = data
        self.event_id = event_id or str(uuid4())
        self.event_type = event_type
        self.timestamp = datetime.now()

    def to_sse(self) -> str:
        """Format as SSE string."""
        lines = [f"id: {self.event_id}"]
        if self.event_type != "message":
            lines.append(f"event: {self.event_type}")
        lines.append(f"data: {json.dumps(self.data, default=str)}")
        return "\n".join(lines) + "\n\n"


class EventStream:
    """SSE event producer for real-time updates.

    Manages client subscriptions and broadcasts events to all
    subscribers of a given channel.
    """

    def __init__(self, max_queue_size: int = 1000):
        self._queues: dict[str, dict[str, asyncio.Queue[Event]]] = defaultdict(dict)
        self._max_queue_size = max_queue_size

    def subscribe(self, channel: str) -> tuple[str, asyncio.Queue[Event]]:
        """Subscribe to a channel. Returns (subscription_id, queue)."""
        sub_id = str(uuid4())
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._max_queue_size)
        self._queues[channel][sub_id] = queue
        return sub_id, queue

    def unsubscribe(self, channel: str, sub_id: str) -> None:
        """Unsubscribe from a channel."""
        if channel in self._queues:
            self._queues[channel].pop(sub_id, None)

    async def publish(self, channel: str, event: Event) -> None:
        """Publish an event to all subscribers of a channel."""
        if channel not in self._queues:
            return

        dead_subs: list[str] = []
        for sub_id, queue in self._queues[channel].items():
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Queue full — drop oldest events
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    dead_subs.append(sub_id)

        # Clean up dead subscriptions
        for sub_id in dead_subs:
            self._queues[channel].pop(sub_id, None)

    async def stream(
        self,
        channel: str,
        heartbeat_interval: float = 15.0,
    ) -> AsyncIterator[str]:
        """Stream events from a channel as SSE strings.

        Args:
            channel: Channel to subscribe to.
            heartbeat_interval: Seconds between heartbeats.

        Yields:
            SSE-formatted strings.
        """
        sub_id, queue = self.subscribe(channel)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=heartbeat_interval,
                    )
                    yield event.to_sse()
                except asyncio.TimeoutError:
                    # Send heartbeat comment to keep connection alive
                    yield f": heartbeat {datetime.now().isoformat()}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            self.unsubscribe(channel, sub_id)

    async def publish_metric(
        self,
        run_id: int,
        metric: MetricRecord,
    ) -> None:
        """Publish a training metric event."""
        event = Event(
            channel=f"metrics:{run_id}",
            data={
                "type": "metric",
                "run_id": run_id,
                "step": metric.step,
                "epoch": metric.epoch,
                "name": metric.metric_name,
                "value": metric.metric_value,
                "timestamp": metric.recorded_at.isoformat()
                if hasattr(metric.recorded_at, "isoformat")
                else str(metric.recorded_at),
            },
        )
        await self.publish(f"metrics:{run_id}", event)
        # Also publish to global metrics channel
        await self.publish("metrics", event)

    async def publish_log(
        self,
        run_id: int,
        line: str,
        level: str = "info",
    ) -> None:
        """Publish a log line event."""
        event = Event(
            channel=f"logs:{run_id}",
            data={
                "type": "log",
                "run_id": run_id,
                "line": line,
                "level": level,
                "timestamp": datetime.now().isoformat(),
            },
        )
        await self.publish(f"logs:{run_id}", event)

    async def publish_pipeline_status(
        self,
        run_id: int,
        status: str,
        script_name: str,
    ) -> None:
        """Publish pipeline status change."""
        event = Event(
            channel="pipeline",
            data={
                "type": "status",
                "run_id": run_id,
                "status": status,
                "script_name": script_name,
                "timestamp": datetime.now().isoformat(),
            },
        )
        await self.publish("pipeline", event)

    async def publish_anomaly(
        self,
        run_id: int,
        alert_type: str,
        severity: str,
        details: dict[str, Any],
    ) -> None:
        """Publish an anomaly detection event."""
        event = Event(
            channel="anomaly",
            data={
                "type": "anomaly",
                "run_id": run_id,
                "alert_type": alert_type,
                "severity": severity,
                "details": details,
                "timestamp": datetime.now().isoformat(),
            },
        )
        await self.publish("anomaly", event)

    async def publish_prompt(
        self,
        run_id: int,
        prompt_text: str,
        domain: str,
        routing_weights: dict[str, float],
        anomaly_detected: bool,
    ) -> None:
        """Publish a prompt analysis event."""
        event = Event(
            channel="prompts",
            data={
                "type": "prompt",
                "run_id": run_id,
                "prompt_text": prompt_text[:200],  # Truncate for SSE
                "domain": domain,
                "routing_weights": routing_weights,
                "anomaly_detected": anomaly_detected,
                "timestamp": datetime.now().isoformat(),
            },
        )
        await self.publish("prompts", event)
