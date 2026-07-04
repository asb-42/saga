#!/usr/bin/env python3
"""
scripts/99_demo_training.py

Demo script that simulates training and emits metrics for UI testing.
Outputs JSON metrics to stdout for the backend to parse.
"""
import argparse
import json
import math
import random
import sys
import time


def emit_metric(name: str, value: float, step: int, epoch: int | None = None):
    """Emit a metric as JSON to stdout."""
    metric = {
        "type": "metric",
        "name": name,
        "value": value,
        "step": step,
        "epoch": epoch,
        "timestamp": time.time(),
    }
    print(json.dumps(metric), flush=True)


def emit_log(line: str, level: str = "info"):
    """Emit a log line as JSON to stdout."""
    log = {
        "type": "log",
        "line": line,
        "level": level,
        "timestamp": time.time(),
    }
    print(json.dumps(log), flush=True)


def simulate_training(epochs: int = 10, steps_per_epoch: int = 50):
    """Simulate training with realistic metrics."""
    emit_log("Starting demo training simulation...", "info")

    for epoch in range(epochs):
        emit_log(f"Epoch {epoch + 1}/{epochs}", "info")

        for step in range(steps_per_epoch):
            global_step = epoch * steps_per_epoch + step

            # Simulate decreasing loss with noise
            base_loss = 0.5 * math.exp(-global_step / 200) + 0.01
            noise = random.gauss(0, 0.005)
            loss = max(0.001, base_loss + noise)

            # Simulate increasing accuracy
            base_acc = 0.5 * (1 - math.exp(-global_step / 150))
            acc_noise = random.gauss(0, 0.01)
            accuracy = min(0.99, max(0.0, base_acc + acc_noise))

            # Simulate learning rate with cosine annealing
            lr = 1e-4 * (0.5 * (1 + math.cos(math.pi * step / steps_per_epoch)))

            # Emit metrics
            emit_metric("train/loss", loss, global_step, epoch)
            emit_metric("train/accuracy", accuracy, global_step, epoch)
            emit_metric("train/lr", lr, global_step, epoch)
            emit_metric("train/epoch", epoch, global_step, epoch)

            # Log every 10 steps
            if step % 10 == 0:
                emit_log(
                    f"[E{epoch + 1}|step {global_step}] loss={loss:.4f} acc={accuracy:.4f} lr={lr:.2e}",
                    "info"
                )

            time.sleep(0.05)  # Simulate computation time

        # Validation at end of epoch
        val_loss = 0.5 * math.exp(-(epoch + 1) * steps_per_epoch / 200) + 0.01
        val_acc = 0.5 * (1 - math.exp(-(epoch + 1) * steps_per_epoch / 150))

        emit_metric("val/loss", val_loss, (epoch + 1) * steps_per_epoch, epoch)
        emit_metric("val/accuracy", val_acc, (epoch + 1) * steps_per_epoch, epoch)
        emit_log(f"Epoch {epoch + 1} complete: val_loss={val_loss:.4f} val_acc={val_acc:.4f}", "info")

    emit_log("Training complete!", "info")


def main():
    parser = argparse.ArgumentParser(description="Demo training script for UI testing")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=50, help="Steps per epoch")
    args = parser.parse_args()

    print("=" * 60)
    print("  SAGA Demo Training — UI Test Script")
    print("=" * 60)

    simulate_training(args.epochs, args.steps_per_epoch)

    print("=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
