"""
src/utils/logging_utils.py

Local-only logging via TensorBoard and optional MLflow.
No external service dependency required.
"""
from __future__ import annotations
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional


class LocalLogger:
    """
    Writes to TensorBoard (always) and MLflow (optional, self-hostable).

    Usage:
        logger = LocalLogger(tb_dir="runs/alignment", run_name="v1")
        logger.log({"loss": 0.42, "lr": 3e-4}, step=100)
        logger.close()
    """

    def __init__(self, tb_dir: str, run_name: str, use_mlflow: bool = False):
        self.writer = SummaryWriter(log_dir=f"{tb_dir}/{run_name}")
        self.use_mlflow = use_mlflow
        if use_mlflow:
            import mlflow
            mlflow.start_run(run_name=run_name)

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step)
        if self.use_mlflow:
            import mlflow
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                step=step,
            )

    def close(self) -> None:
        self.writer.close()
        if self.use_mlflow:
            import mlflow
            mlflow.end_run()
