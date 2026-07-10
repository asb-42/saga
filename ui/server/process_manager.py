"""Process manager for SAGA pipeline scripts."""
from __future__ import annotations

import asyncio
import os
import signal
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import config
from .event_stream import Event, EventStream
from .metric_collector import MetricCollector
from .models import ScriptRun, ScriptStatus
from .script_params import SCRIPT_FILE_MAP
from .storage import Storage


class ProcessManager:
    """Manages lifecycle of SAGA pipeline scripts.

    Scripts run as independent subprocesses for isolation.
    Supports start, pause (SIGSTOP), resume (SIGCONT), and stop (SIGTERM).
    """

    def __init__(
        self,
        storage: Storage,
        event_stream: EventStream,
    ):
        self.storage = storage
        self.events = event_stream
        self.metrics = MetricCollector(storage, event_stream)
        self._processes: dict[int, asyncio.subprocess.Process] = {}
        self._tasks: dict[int, asyncio.Task] = {}

    async def start(
        self,
        script_name: str,
        parameters: dict[str, Any] | None = None,
        source: str = "ui",
    ) -> ScriptRun:
        """Start a script as a subprocess.

        Args:
            script_name: Script name (e.g., '02_train_alignment').
            parameters: CLI arguments to pass to the script.
            source: How the run was initiated ('ui', 'cli').

        Returns:
            The created ScriptRun record.
        """
        params = parameters or {}

        # Create database record
        run = await self.storage.create_run(script_name, params, source=source)
        await self.storage.update_run_status(run.id, ScriptStatus.RUNNING)
        await self.events.publish_pipeline_status(
            run.id, "running", script_name
        )

        # Build command — resolve UI script ID to actual filename
        script_filename = SCRIPT_FILE_MAP.get(script_name, f"{script_name}.py")
        script_path = config.SCRIPTS_DIR / script_filename
        if not script_path.exists():
            error_msg = f"Script not found: {script_path}"
            await self.storage.update_run_status(
                run.id, ScriptStatus.FAILED, exit_code=-1
            )
            await self.storage.set_run_error(run.id, error_msg)
            raise FileNotFoundError(error_msg)

        cmd = [config.PYTHON_EXECUTABLE, str(script_path)]
        for key, value in params.items():
            cmd.append(f"--{key}")
            if isinstance(value, list):
                cmd.extend(str(v) for v in value)
            else:
                cmd.append(str(value))

        # Start subprocess — use start_new_session=True so SIGTERM/SIGKILL
        # propagates to child processes (Python, torch, etc.)
        # PYTHONUNBUFFERED=1 ensures stdout/stderr are not buffered
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(config.PROJECT_ROOT),
            start_new_session=True,
            env=env,
        )
        self._processes[run.id] = process

        # Start output monitoring task
        task = asyncio.create_task(
            self._monitor_process(run.id, process, script_name)
        )
        self._tasks[run.id] = task

        # Update record with PID info
        run.status = ScriptStatus.RUNNING
        return run

    async def pause(self, run_id: int) -> None:
        """Pause a running script (SIGSTOP)."""
        process = self._processes.get(run_id)
        if process and process.returncode is None:
            os.kill(process.pid, signal.SIGSTOP)
            await self.storage.update_run_status(run_id, ScriptStatus.PAUSED)
            await self.events.publish_pipeline_status(
                run_id, "paused", ""
            )

    async def resume(self, run_id: int) -> None:
        """Resume a paused script (SIGCONT)."""
        process = self._processes.get(run_id)
        if process and process.returncode is None:
            os.kill(process.pid, signal.SIGCONT)
            await self.storage.update_run_status(run_id, ScriptStatus.RUNNING)
            await self.events.publish_pipeline_status(
                run_id, "running", ""
            )

    async def stop(self, run_id: int, timeout: float = 5.0) -> bool:
        """Stop a script (SIGTERM, then SIGKILL if needed).

        Args:
            run_id: Run ID to stop.
            timeout: Seconds to wait before SIGKILL.

        Returns:
            True if stop was successful, False if process not found/stale.
        """
        process = self._processes.get(run_id)
        if not process:
            # Process not in live table — check if DB says it's still running
            run = await self.storage.get_run(run_id)
            if run and run.status == ScriptStatus.RUNNING:
                # Stale status: process exited but monitor didn't clean up DB
                await self.storage.update_run_status(
                    run_id, ScriptStatus.FAILED, exit_code=-1
                )
                await self.events.publish_pipeline_status(
                    run_id, "failed", ""
                )
            return False

        if process.returncode is not None:
            # Already finished
            return False

        # Try graceful shutdown
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Force kill
            process.kill()
            await process.wait()

        # Cancel monitoring task
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()

        exit_code = process.returncode or -1
        status = (
            ScriptStatus.COMPLETED if exit_code == 0 else ScriptStatus.FAILED
        )
        await self.storage.update_run_status(run_id, status, exit_code)
        await self.events.publish_pipeline_status(
            run_id, status.value, ""
        )
        return True

    async def get_status(self, run_id: int) -> ScriptStatus | None:
        """Get current status of a script."""
        run = await self.storage.get_run(run_id)
        return run.status if run else None

    async def list_runs(
        self,
        status: ScriptStatus | None = None,
        script_name: str | None = None,
        limit: int = 100,
    ) -> list[ScriptRun]:
        """List all runs with optional filters."""
        return await self.storage.list_runs(status=status, script_name=script_name, limit=limit)

    async def shutdown(self) -> None:
        """Stop all running processes and cancel monitoring tasks."""
        # Cancel all monitoring tasks
        for task in self._tasks.values():
            if not task.done():
                task.cancel()

        # Wait for tasks to finish cancellation
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        # Stop all running processes
        for run_id, process in list(self._processes.items()):
            if process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass

        self._processes.clear()
        self._tasks.clear()

    async def _monitor_process(
        self,
        run_id: int,
        process: asyncio.subprocess.Process,
        script_name: str,
    ) -> None:
        """Monitor a subprocess, streaming output and handling completion."""
        last_lines: list[str] = []
        max_last_lines = 50

        try:
            assert process.stdout is not None
            assert process.stderr is not None

            # Read stdout: publish as info logs and collect last lines
            async def read_stdout(stream):
                async for line in stream:
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        await self.metrics.process_line(run_id, text)
                        await self.events.publish_log(run_id, text, "info")
                        # Also publish to global log channel
                        await self.events.publish("logs:all", Event(
                            channel="logs:all",
                            data={
                                "type": "log",
                                "run_id": run_id,
                                "line": text,
                                "level": "info",
                                "script_name": script_name,
                                "timestamp": datetime.now().isoformat(),
                            },
                        ))
                        last_lines.append(text)
                        if len(last_lines) > max_last_lines:
                            last_lines.pop(0)

            # Read stderr: publish as error logs and collect last lines
            async def read_stderr(stream):
                async for line in stream:
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        await self.events.publish_log(run_id, text, "error")
                        # Also publish to global log channel
                        await self.events.publish("logs:all", Event(
                            channel="logs:all",
                            data={
                                "type": "log",
                                "run_id": run_id,
                                "line": text,
                                "level": "error",
                                "script_name": script_name,
                                "timestamp": datetime.now().isoformat(),
                            },
                        ))
                        last_lines.append(text)
                        if len(last_lines) > max_last_lines:
                            last_lines.pop(0)

            await asyncio.gather(
                read_stdout(process.stdout),
                read_stderr(process.stderr),
            )

            # Wait for process to complete
            exit_code = await process.wait()

            # Store last output in run record
            await self.storage.append_run_output(run_id, "\n".join(last_lines[-max_last_lines:]))

            # Update status
            status = (
                ScriptStatus.COMPLETED if exit_code == 0 else ScriptStatus.FAILED
            )
            await self.storage.update_run_status(run_id, status, exit_code)
            await self.events.publish_pipeline_status(
                run_id, status.value, script_name
            )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            await self.storage.update_run_status(
                run_id, ScriptStatus.FAILED, exit_code=-1
            )
            await self.storage.set_run_error(run_id, str(e))
            await self.events.publish_log(
                run_id, f"Monitor error: {e}", "error"
            )
        finally:
            self._processes.pop(run_id, None)
            self._tasks.pop(run_id, None)
