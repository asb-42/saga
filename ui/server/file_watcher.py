"""File watcher for monitoring log files in real-time."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from watchdog.observers import Observer

from .event_stream import EventStream


class LogFileHandler(FileSystemEventHandler):
    """Handles file modification events for log files."""

    def __init__(self, callback: Callable[[str, str], None]):
        self.callback = callback
        self._file_positions: dict[str, int] = {}

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification."""
        if event.is_directory:
            return

        file_path = event.src_path
        if not any(file_path.endswith(ext) for ext in [".log", ".txt", ".jsonl"]):
            return

        try:
            # Read only new content
            last_pos = self._file_positions.get(file_path, 0)
            with open(file_path, "r", errors="replace") as f:
                f.seek(last_pos)
                new_content = f.read()
                self._file_positions[file_path] = f.tell()

            if new_content.strip():
                self.callback(file_path, new_content)
        except Exception:
            pass


class FileWatcher:
    """Watches directories for log file changes and streams them via SSE.

    Args:
        event_stream: SSE event stream for broadcasting.
        watch_dirs: Directories to watch for log files.
    """

    def __init__(
        self,
        event_stream: EventStream,
        watch_dirs: list[str | Path] | None = None,
    ):
        self.events = event_stream
        self._observer: Observer | None = None
        self._watch_dirs = [Path(d) for d in (watch_dirs or [])]
        self._loop: asyncio.AbstractEventLoop | None = None

    def add_watch_dir(self, path: str | Path) -> None:
        """Add a directory to watch."""
        self._watch_dirs.append(Path(path))

    def start(self) -> None:
        """Start watching for file changes."""
        if self._observer:
            return

        self._loop = asyncio.get_event_loop()
        self._observer = Observer()

        for watch_dir in self._watch_dirs:
            if watch_dir.exists():
                handler = LogFileHandler(self._on_file_changed)
                self._observer.schedule(handler, str(watch_dir), recursive=True)

        self._observer.start()

    def stop(self) -> None:
        """Stop watching."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    def _on_file_changed(self, file_path: str, content: str) -> None:
        """Handle file changes by broadcasting via SSE."""
        if not self._loop:
            return

        # Extract meaningful info from content
        lines = content.strip().split("\n")
        for line in lines:
            if line.strip():
                # Determine log level
                level = "info"
                lower_line = line.lower()
                if "error" in lower_line or "traceback" in lower_line:
                    level = "error"
                elif "warning" in lower_line:
                    level = "warning"

                # Schedule coroutine on main event loop from watchdog thread
                asyncio.run_coroutine_threadsafe(
                    self.events.publish_log(
                        run_id=0,  # Generic log, not tied to a specific run
                        line=f"[{Path(file_path).name}] {line.strip()}",
                        level=level,
                    ),
                    self._loop,
                )
