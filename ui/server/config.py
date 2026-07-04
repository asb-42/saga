"""Server configuration."""
from __future__ import annotations

from pathlib import Path


class Config:
    """Server configuration with sensible defaults."""

    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
    CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    LOGS_DIR: Path = PROJECT_ROOT / "ui" / "logs"

    # Database
    DATABASE_URL: str = f"sqlite+aiosqlite:///{PROJECT_ROOT / 'ui' / 'saga_lab.db'}"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8420
    DEBUG: bool = False

    # Process Management
    PYTHON_EXECUTABLE: str = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    SCRIPT_TIMEOUT: int = 3600  # 1 hour default

    # SSE
    SSE_HEARTBEAT_INTERVAL: int = 15  # seconds
    SSE_MAX_QUEUE_SIZE: int = 1000

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create required directories if they don't exist."""
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
