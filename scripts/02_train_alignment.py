#!/usr/bin/env python3
"""
scripts/02_train_alignment.py

Launch embedding alignment training.
Simply calls train_alignment() from src.alignment.trainer.
"""
import argparse
import sys
from pathlib import Path

# ── add project root to path ────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.alignment.trainer import train_alignment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Train embedding alignment projectors (InfoNCE)"
    )
    parser.add_argument(
        "--config", default="configs/alignment.yaml",
        help="Alignment config path",
    )
    parser.add_argument(
        "--models-config", default="configs/models.yaml",
        help="Models config path",
    )
    parser.add_argument(
        "--output-dir", default="checkpoints/alignment",
        help="Output directory (overrides config)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SAGA — Embedding Alignment Training")
    print(f"  Config:  {args.config}")
    print(f"  Models:  {args.models_config}")
    print(f"  Output:  {args.output_dir}")
    print("=" * 60)

    sys.exit(
        train_alignment(
            config_path=args.config,
            models_config_path=args.models_config,
        )
    )


if __name__ == "__main__":
    main()
