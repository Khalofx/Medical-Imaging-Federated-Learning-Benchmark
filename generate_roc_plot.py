#!/usr/bin/env python3
"""
Generate ROC curve panels from the saved best checkpoints.

Example:
    python generate_roc_plot.py
    python generate_roc_plot.py --device cuda --num-workers 8
"""

import argparse
import json
from pathlib import Path

from visualize import plot_test_roc_curves


def main():
    parser = argparse.ArgumentParser(
        description="Generate test ROC curves from saved best FL checkpoints."
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("results/completed_experiments.json"),
        help="Path to completed benchmark registry JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Directory where roc_curves.png will be saved.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing per-algorithm best checkpoints.",
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=Path("results/data_splits.pt"),
        help="Path to saved benchmark data splits.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for forward passes, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader worker override for ROC generation.",
    )
    parser.add_argument(
        "--algos",
        nargs="*",
        default=None,
        help="Optional list of algorithms to include.",
    )
    parser.add_argument(
        "--alphas",
        nargs="*",
        type=float,
        default=None,
        help="Optional list of alpha values to include.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny validation pass on the first available FL result only.",
    )
    args = parser.parse_args()

    with open(args.results_json) as f:
        all_results = list(json.load(f).values())

    if args.algos is not None:
        all_results = [r for r in all_results if r["algo"] in set(args.algos)]
    if args.alphas is not None:
        keep = set(args.alphas)
        all_results = [r for r in all_results if r["alpha"] in keep]
    if args.smoke:
        all_results = [
            r for r in all_results if r["algo"] not in ("Central", "Local-Only", "DP-FedAvg")
        ][:1]

    plot_test_roc_curves(
        all_results=all_results,
        out_dir=args.out_dir,
        checkpoint_dir=str(args.checkpoint_dir),
        device=args.device,
        num_workers=args.num_workers,
        splits_path=str(args.splits_path),
    )


if __name__ == "__main__":
    main()
