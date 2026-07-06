#!/usr/bin/env python3
"""
scripts/02b_train_alignment_structured.py

Train embedding alignment with structure preservation loss.
Combines InfoNCE (align identical prompts) with structure preservation
(preserves relative distances between prompts).

Usage:
  python scripts/02b_train_alignment_structured.py \\
    --config configs/alignment_structured.yaml \\
    --output-dir checkpoints/alignment_structured
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
        description="Train alignment with structure preservation loss"
    )
    parser.add_argument(
        "--config", default="configs/alignment_structured.yaml",
        help="Alignment config path",
    )
    parser.add_argument(
        "--models-config", default="configs/models.yaml",
        help="Models config path",
    )
    parser.add_argument(
        "--output-dir", default="checkpoints/alignment_structured",
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--structure-weight", type=float, default=0.1,
        help="Structure preservation loss weight (λ)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SAGA — Structured Alignment Training")
    print(f"  Config:  {args.config}")
    print(f"  Models:  {args.models_config}")
    print(f"  Output:  {args.output_dir}")
    print(f"  λ:       {args.structure_weight}")
    print("=" * 60)

    sys.exit(
        train_alignment(
            config_path=args.config,
            models_config_path=args.models_config,
            output_dir_override=args.output_dir,
            structure_weight_override=args.structure_weight,
        )
    )


if __name__ == "__main__":
    main()
