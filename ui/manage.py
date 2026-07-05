#!/usr/bin/env python3
"""
SAGA Research Lab — UI Management Script.

Start/stop backend and frontend, view logs, all from one terminal.

Usage:
    python ui/manage.py start
    python ui/manage.py stop
    python ui/manage.py status
    python ui/manage.py logs [backend|frontend]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PID_FILE = PROJECT_ROOT / "ui" / ".ui_processes.json"
LOG_DIR = PROJECT_ROOT / "ui" / "logs"

BACKEND_CMD = [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-m", "server.main"]
BACKEND_DIR = PROJECT_ROOT / "ui"
FRONTEND_CMD = ["npm", "run", "dev"]
FRONTEND_DIR = PROJECT_ROOT / "ui" / "frontend"


def ensure_dirs():
    LOG_DIR.mkdir(exist_ok=True)


def load_pids() -> dict:
    if PID_FILE.exists():
        with open(PID_FILE) as f:
            return json.load(f)
    return {}


def save_pids(pids: dict):
    with open(PID_FILE, "w") as f:
        json.dump(pids, f)


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def start_backend() -> int | None:
    """Start the backend server."""
    log_file = LOG_DIR / "backend.log"
    proc = subprocess.Popen(
        BACKEND_CMD,
        cwd=str(BACKEND_DIR),
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc.pid


def start_frontend() -> int | None:
    """Start the frontend dev server."""
    log_file = LOG_DIR / "frontend.log"
    proc = subprocess.Popen(
        FRONTEND_CMD,
        cwd=str(FRONTEND_DIR),
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc.pid


def stop_process(pid: int) -> bool:
    """Stop a process by PID."""
    if not is_running(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait up to 5 seconds
        for _ in range(50):
            if not is_running(pid):
                return True
            time.sleep(0.1)
        # Force kill
        os.kill(pid, signal.SIGKILL)
        return True
    except (OSError, ProcessLookupError):
        return True


def cmd_start(args):
    ensure_dirs()
    pids = load_pids()

    if args.backend or args.all:
        if "backend" in pids and is_running(pids["backend"]):
            print(f"⚠️  Backend already running (PID {pids['backend']})")
        else:
            pid = start_backend()
            if pid:
                pids["backend"] = pid
                print(f"✓ Backend started (PID {pid})")
                print(f"  URL: http://localhost:8420")
                print(f"  Logs: {LOG_DIR / 'backend.log'}")

    if args.frontend or args.all:
        if "frontend" in pids and is_running(pids["frontend"]):
            print(f"⚠️  Frontend already running (PID {pids['frontend']})")
        else:
            pid = start_frontend()
            if pid:
                pids["frontend"] = pid
                print(f"✓ Frontend started (PID {pid})")
                print(f"  URL: http://localhost:5173")
                print(f"  Logs: {LOG_DIR / 'frontend.log'}")

    save_pids(pids)

    if not args.backend and not args.frontend and not args.all:
        # Start both
        if "backend" not in pids or not is_running(pids.get("backend", 0)):
            pid = start_backend()
            if pid:
                pids["backend"] = pid
                print(f"✓ Backend started (PID {pid})")
                print(f"  URL: http://localhost:8420")

        if "frontend" not in pids or not is_running(pids.get("frontend", 0)):
            pid = start_frontend()
            if pid:
                pids["frontend"] = pid
                print(f"✓ Frontend started (PID {pid})")
                print(f"  URL: http://localhost:5173")

        save_pids(pids)


def cmd_stop(args):
    pids = load_pids()

    if args.backend or args.all or (not args.backend and not args.frontend):
        if "backend" in pids:
            if stop_process(pids["backend"]):
                print(f"✓ Backend stopped")
                del pids["backend"]
        else:
            print("ℹ️  Backend not running")

    if args.frontend or args.all or (not args.backend and not args.frontend):
        if "frontend" in pids:
            if stop_process(pids["frontend"]):
                print(f"✓ Frontend stopped")
                del pids["frontend"]
        else:
            print("ℹ️  Frontend not running")

    save_pids(pids)


def cmd_status(args):
    pids = load_pids()

    print("=== SAGA Research Lab ===\n")

    for name in ["backend", "frontend"]:
        pid = pids.get(name)
        if pid and is_running(pid):
            print(f"  {name:10} 🟢 Running (PID {pid})")
        elif pid:
            print(f"  {name:10} 🔴 Stopped (PID {pid} dead)")
        else:
            print(f"  {name:10} ⚪ Not started")

    print(f"\n  Logs: {LOG_DIR}")


def cmd_logs(args):
    target = args.target or "backend"
    log_file = LOG_DIR / f"{target}.log"

    if not log_file.exists():
        print(f"⚠️  No log file for {target}")
        return

    if args.follow:
        # Follow mode (like tail -f)
        print(f"Following {target} logs (Ctrl+C to stop)...\n")
        try:
            with open(log_file) as f:
                # Go to end
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if line:
                        print(line.rstrip())
                    else:
                        time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped following.")
    else:
        # Show last N lines
        lines = args.lines or 50
        with open(log_file) as f:
            all_lines = f.readlines()
            for line in all_lines[-lines:]:
                print(line.rstrip())


def cmd_restart(args):
    cmd_stop(args)
    time.sleep(1)
    cmd_start(args)


def main():
    parser = argparse.ArgumentParser(
        description="SAGA Research Lab UI Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start          Start backend + frontend
  %(prog)s start -b       Start backend only
  %(prog)s start -f       Start frontend only
  %(prog)s stop           Stop everything
  %(prog)s status         Show running status
  %(prog)s logs backend   Show backend logs
  %(prog)s logs frontend -f   Follow frontend logs
  %(prog)s restart        Restart everything
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # start
    start_p = sub.add_parser("start", help="Start services")
    start_p.add_argument("-b", "--backend", action="store_true", help="Start backend only")
    start_p.add_argument("-f", "--frontend", action="store_true", help="Start frontend only")
    start_p.add_argument("-a", "--all", action="store_true", help="Start both")

    # stop
    stop_p = sub.add_parser("stop", help="Stop services")
    stop_p.add_argument("-b", "--backend", action="store_true", help="Stop backend only")
    stop_p.add_argument("-f", "--frontend", action="store_true", help="Stop frontend only")
    stop_p.add_argument("-a", "--all", action="store_true", help="Stop both")

    # status
    sub.add_parser("status", help="Show service status")

    # logs
    logs_p = sub.add_parser("logs", help="View logs")
    logs_p.add_argument("target", nargs="?", default="backend", choices=["backend", "frontend"], help="Which logs to view")
    logs_p.add_argument("-f", "--follow", action="store_true", help="Follow log output")
    logs_p.add_argument("-n", "--lines", type=int, default=50, help="Number of lines to show")

    # restart
    restart_p = sub.add_parser("restart", help="Restart services")
    restart_p.add_argument("-b", "--backend", action="store_true", help="Restart backend only")
    restart_p.add_argument("-f", "--frontend", action="store_true", help="Restart frontend only")
    restart_p.add_argument("-a", "--all", action="store_true", help="Restart both")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "logs": cmd_logs,
        "restart": cmd_restart,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
