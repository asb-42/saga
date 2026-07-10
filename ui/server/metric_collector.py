"""Metric collector for parsing script output and storing metrics."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from .event_stream import EventStream
from .models import MetricRecord
from .storage import Storage


class MetricCollector:
    """Collects metrics from script stdout and stores them.

    Parses JSON metric lines from script output and:
    1. Stores in SQLite database
    2. Broadcasts via SSE event stream
    """

    def __init__(self, storage: Storage, event_stream: EventStream):
        self.storage = storage
        self.events = event_stream

    async def process_line(self, run_id: int, line: str) -> None:
        """Process a single line of script output.

        Detects JSON metric/log lines and processes them.
        Non-JSON lines are ignored (treated as plain logs).
        """
        if not line or not line.strip():
            return

        line = line.strip()

        # Try to parse as JSON metric
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                msg_type = data.get("type")

                if msg_type == "metric":
                    await self._handle_metric(run_id, data)
                elif msg_type == "log":
                    await self._handle_log(run_id, data)
                elif msg_type == "prompt_result":
                    await self._handle_eval_progress(run_id, data)
                elif msg_type in ("alignment_progress", "alignment_epoch", "alignment_start"):
                    await self._handle_alignment_progress(run_id, data)
                elif msg_type in ("oracle_progress", "oracle_start", "oracle_complete"):
                    await self._handle_oracle_progress(run_id, data)
                elif msg_type in ("router_train_start", "router_train_step", "router_train_epoch", "router_train_complete"):
                    await self._handle_router_train_progress(run_id, data)
                else:
                    # Unknown JSON type, treat as log
                    await self.events.publish_log(run_id, line, "info")
                return
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to parse TensorBoard-style log line
        # Format: [E1|step 100] loss=0.0420 acc=0.8500 lr=1.00e-04
        metric_match = re.search(
            r'\[E(\d+)\|step (\d+)\]\s+'
            r'(?:loss=([0-9.]+)\s+)?'
            r'(?:acc=([0-9.]+)\s+)?'
            r'(?:lr=([0-9.e-]+)\s*)?',
            line
        )
        if metric_match:
            epoch = int(metric_match.group(1)) - 1
            step = int(metric_match.group(2))

            if metric_match.group(3):
                await self._store_and_broadcast(
                    run_id, "train/loss", float(metric_match.group(3)), step, epoch
                )
            if metric_match.group(4):
                await self._store_and_broadcast(
                    run_id, "train/accuracy", float(metric_match.group(4)), step, epoch
                )
            if metric_match.group(5):
                await self._store_and_broadcast(
                    run_id, "train/lr", float(metric_match.group(5)), step, epoch
                )

        # Publish as plain log
        await self.events.publish_log(run_id, line, "info")

    async def _handle_metric(self, run_id: int, data: dict[str, Any]) -> None:
        """Handle a JSON metric line."""
        name = data.get("name", "")
        value = data.get("value", 0.0)
        step = data.get("step", 0)
        epoch = data.get("epoch")

        await self._store_and_broadcast(run_id, name, value, step, epoch)

    async def _handle_log(self, run_id: int, data: dict[str, Any]) -> None:
        """Handle a JSON log line."""
        line = data.get("line", "")
        level = data.get("level", "info")
        await self.events.publish_log(run_id, line, level)

    async def _handle_eval_progress(self, run_id: int, data: dict[str, Any]) -> None:
        """Handle eval progress / per-prompt result events."""
        from .event_stream import Event

        # Publish to eval:progress channel (for eval monitor page)
        await self.events.publish("eval:progress", Event(
            channel="eval:progress",
            data={**data, "run_id": run_id},
        ))

        # Also publish prompt_result to prompts channel (for live feed page)
        if data.get("type") == "prompt_result":
            await self.events.publish_prompt(
                run_id=run_id,
                prompt_text=data.get("prompt", ""),
                domain="nl",
                routing_weights={data.get("model", "unknown"): 1.0},
                anomaly_detected=False,
                passed=data.get("passed"),
                benchmark=data.get("benchmark"),
            )

    async def _handle_alignment_progress(self, run_id: int, data: dict[str, Any]) -> None:
        """Handle alignment training progress events (start, step, epoch)."""
        from .event_stream import Event

        # Publish to alignment:progress channel (for alignment monitor page)
        await self.events.publish("alignment:progress", Event(
            channel="alignment:progress",
            data={**data, "run_id": run_id},
        ))

    async def _handle_oracle_progress(self, run_id: int, data: dict[str, Any]) -> None:
        """Handle oracle label generation progress events."""
        from .event_stream import Event

        await self.events.publish("oracle:progress", Event(
            channel="oracle:progress",
            data={**data, "run_id": run_id},
        ))

    async def _handle_router_train_progress(self, run_id: int, data: dict[str, Any]) -> None:
        """Handle router training progress events."""
        from .event_stream import Event

        await self.events.publish("router:progress", Event(
            channel="router:progress",
            data={**data, "run_id": run_id},
        ))

    async def _store_and_broadcast(
        self,
        run_id: int,
        name: str,
        value: float,
        step: int,
        epoch: int | None = None,
    ) -> None:
        """Store metric in DB and broadcast via SSE."""
        # Store in database
        metric = await self.storage.store_metric(
            run_id=run_id,
            name=name,
            value=value,
            step=step,
            epoch=epoch,
        )

        # Broadcast via SSE
        await self.events.publish_metric(run_id, metric)
