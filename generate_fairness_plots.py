#!/usr/bin/env python3
"""
Generate hospital-fairness plots by evaluating saved best checkpoints on
scanner-specific test sets.

Fairness notion used here:
  performance consistency across hospitals / scanner domains

Outputs:
  - fairness_hospital_metrics.csv
  - fairness_auc_gap_heatmap.png
  - fairness_hospital_auc.png
  - fairness_best_worst_hospitals.csv
  - fairness_best_worst_auc.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA, FED, MODEL
from dataset import CXRBinaryDataset, base_transforms, build_image_index, get_pos_weight
from evaluate import evaluate_model
from models import build_vit


COLORS = {
    "FedAvg": "#4C72B0",
    "FedProx": "#DD8452",
    "SCAFFOLD": "#55A868",
    "FedNova": "#C44E52",
    "Central": "#000000",
}


def _result_key(algo: str, alpha):
    return algo if alpha is None else f"{algo}_alpha{alpha}"


def _scanner_cache_key(alpha, scanner_idx, scanner):
    return (
        alpha,
        scanner_idx,
        round(float(scanner["gamma"]), 4),
        bool(scanner["blur"]),
    )


def _build_test_loader(test_df, image_index, scanner, num_workers):
    ds = CXRBinaryDataset(
        test_df,
        DATA.img_dir,
        transform=base_transforms(DATA.img_size, is_train=False, **scanner),
        image_index=image_index,
    )
    return DataLoader(
        ds,
        batch_size=FED.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def _load_model_state(algo: str, alpha, checkpoint_dir: Path, device: str):
    if algo == "Central":
        ckpt_path = checkpoint_dir / "Central" / "best.pt"
    else:
        ckpt_path = checkpoint_dir / algo / f"alpha{alpha}_best.pt"

    if not ckpt_path.exists():
        return None

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "client_states" in state:
        return None
    return state


def main():
    parser = argparse.ArgumentParser(
        description="Generate fairness plots from saved best checkpoints."
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("results/completed_experiments.json"),
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=Path("results/data_splits.pt"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu or cuda",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader worker override.",
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
        help="Run a tiny validation pass on the first compatible result only.",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = DATA.num_workers if args.num_workers is None else args.num_workers
    out_dir = args.out_dir
    out_dir.mkdir(exist_ok=True)

    with open(args.results_json) as f:
        results = list(json.load(f).values())
    split_store = torch.load(args.splits_path, map_location="cpu", weights_only=False)

    if args.algos is not None:
        results = [r for r in results if r["algo"] in set(args.algos)]
    if args.alphas is not None:
        keep = set(args.alphas)
        results = [r for r in results if r["alpha"] in keep]
    if args.smoke:
        filtered = [
            r for r in results
            if r["algo"] not in ("DP-FedAvg", "Local-Only")
        ]
        results = filtered[:1]

    image_index = build_image_index(DATA.img_dir)
    loader_cache = {}
    rows = []

    any_split = next(iter(split_store.values()))

    for result in results:
        algo = result["algo"]
        alpha = result["alpha"]

        if algo == "DP-FedAvg":
            continue
        if algo == "Local-Only":
            continue

        state = _load_model_state(algo, alpha, args.checkpoint_dir, device)
        if state is None:
            print(f"Skipping fairness eval for {algo} α={alpha}: no compatible checkpoint found.")
            continue

        split_key = _result_key(algo, alpha)
        split = split_store.get(split_key, any_split)
        train_df = split["train_df"]
        test_df = split["test_df"]
        scanners = split["scanners"]
        criterion = nn.BCEWithLogitsLoss(pos_weight=get_pos_weight(train_df).to(device))

        model = build_vit(MODEL.num_classes, pretrained=False).to(device)
        model.load_state_dict(state)

        for client_idx, scanner in enumerate(scanners):
            cache_key = _scanner_cache_key(alpha, client_idx, scanner)
            if cache_key not in loader_cache:
                loader_cache[cache_key] = _build_test_loader(
                    test_df, image_index, scanner, num_workers
                )
            metrics = evaluate_model(model, loader_cache[cache_key], criterion, device)
            rows.append(
                {
                    "algorithm": algo,
                    "alpha": alpha,
                    "client": client_idx,
                    "auc": metrics["auc"],
                    "accuracy": metrics["accuracy"],
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                }
            )
            print(
                f"{algo:>8}  α={alpha!s:<4}  hospital={client_idx}  "
                f"AUC={metrics['auc']:.4f}"
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No fairness rows were generated.")

    df.to_csv(out_dir / "fairness_hospital_metrics.csv", index=False)
    print(f"Saved → {out_dir / 'fairness_hospital_metrics.csv'}")

    best_worst_rows = []
    for (algo, alpha), group in df.groupby(["algorithm", "alpha"], dropna=False):
        best_idx = group["auc"].idxmax()
        worst_idx = group["auc"].idxmin()
        best_row = group.loc[best_idx]
        worst_row = group.loc[worst_idx]
        best_worst_rows.append(
            {
                "algorithm": algo,
                "alpha": alpha,
                "best_hospital": int(best_row["client"]),
                "best_auc": float(best_row["auc"]),
                "worst_hospital": int(worst_row["client"]),
                "worst_auc": float(worst_row["auc"]),
                "auc_gap": float(best_row["auc"] - worst_row["auc"]),
            }
        )

    best_worst_df = pd.DataFrame(best_worst_rows)
    best_worst_df.to_csv(out_dir / "fairness_best_worst_hospitals.csv", index=False)
    print(f"Saved → {out_dir / 'fairness_best_worst_hospitals.csv'}")

    fl_df = df[df["algorithm"] != "Central"].copy()
    algos = []
    alphas = []
    for _, row in fl_df.iterrows():
        if row["algorithm"] not in algos:
            algos.append(row["algorithm"])
        if row["alpha"] not in alphas:
            alphas.append(row["alpha"])
    alphas = sorted(alphas)

    gap_matrix = np.full((len(algos), len(alphas)), np.nan)
    for i, algo in enumerate(algos):
        for j, alpha in enumerate(alphas):
            vals = fl_df[(fl_df["algorithm"] == algo) & (fl_df["alpha"] == alpha)]["auc"]
            if len(vals):
                gap_matrix[i, j] = vals.max() - vals.min()

    central_vals = df[df["algorithm"] == "Central"]["auc"]
    if len(central_vals):
        central_gap = central_vals.max() - central_vals.min()
        gap_matrix = np.vstack([np.full((1, len(alphas)), central_gap), gap_matrix])
        gap_labels = ["── Central ──"] + algos
    else:
        gap_labels = algos

    fig, ax = plt.subplots(figsize=(7, len(gap_labels) * 0.7 + 1.5))
    sns.heatmap(
        gap_matrix,
        annot=True,
        fmt=".3f",
        cmap="mako_r",
        xticklabels=[f"α={a}" for a in alphas],
        yticklabels=gap_labels,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "AUC gap across hospitals"},
    )
    ax.set_title("Hospital Fairness Gap  (lower = more consistent)")
    ax.set_xlabel("Dirichlet α")
    ax.set_ylabel("Method")
    fig.tight_layout()
    fig.savefig(out_dir / "fairness_auc_gap_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_dir / 'fairness_auc_gap_heatmap.png'}")

    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 4), sharey=True)
    if len(alphas) == 1:
        axes = [axes]
    for ax, alpha in zip(axes, alphas):
        panel = fl_df[fl_df["alpha"] == alpha]
        for algo in algos:
            subset = panel[panel["algorithm"] == algo].sort_values("client")
            if subset.empty:
                continue
            ax.plot(
                subset["client"],
                subset["auc"],
                marker="o",
                linewidth=2,
                color=COLORS.get(algo),
                label=algo,
            )
        if len(central_vals):
            central_by_client = df[df["algorithm"] == "Central"].sort_values("client")
            ax.plot(
                central_by_client["client"],
                central_by_client["auc"],
                marker="s",
                linewidth=1.8,
                linestyle="--",
                color=COLORS["Central"],
                label="Central",
            )
        ax.set_title(f"α = {alpha}")
        ax.set_xlabel("Hospital / Client")
        ax.set_ylabel("Test AUC")
        ax.set_xticks(sorted(panel["client"].unique()))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Hospital-wise Test AUC by Method", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fairness_hospital_auc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_dir / 'fairness_hospital_auc.png'}")

    plot_df = best_worst_df[best_worst_df["algorithm"] != "Local-Only"].copy()
    if not plot_df.empty:
        alphas_plot = sorted(a for a in plot_df["alpha"].dropna().unique())
        fig, axes = plt.subplots(
            1, len(alphas_plot), figsize=(5 * len(alphas_plot), 4), sharey=True
        )
        if len(alphas_plot) == 1:
            axes = [axes]

        for ax, alpha in zip(axes, alphas_plot):
            panel = plot_df[plot_df["alpha"] == alpha].copy()
            if panel.empty:
                continue
            panel = panel.sort_values("best_auc", ascending=False).reset_index(drop=True)
            y = np.arange(len(panel))
            ax.barh(
                y,
                panel["best_auc"] - panel["worst_auc"],
                left=panel["worst_auc"],
                color=[COLORS.get(algo, "#888888") for algo in panel["algorithm"]],
                alpha=0.35,
                edgecolor="none",
            )
            ax.scatter(
                panel["worst_auc"],
                y,
                color="#C44E52",
                marker="o",
                s=55,
                label="Worst hospital",
            )
            ax.scatter(
                panel["best_auc"],
                y,
                color="#55A868",
                marker="o",
                s=55,
                label="Best hospital",
            )

            for yi, row in enumerate(panel.itertuples(index=False)):
                ax.text(
                    row.worst_auc - 0.003,
                    yi,
                    f"H{row.worst_hospital}",
                    va="center",
                    ha="right",
                    fontsize=8,
                )
                ax.text(
                    row.best_auc + 0.003,
                    yi,
                    f"H{row.best_hospital}",
                    va="center",
                    ha="left",
                    fontsize=8,
                )

            ax.set_title(f"α = {alpha}")
            ax.set_xlabel("Test AUC")
            ax.set_yticks(y)
            ax.set_yticklabels(panel["algorithm"])
            ax.grid(True, axis="x", alpha=0.3)
            if ax is axes[0]:
                ax.legend(fontsize=8, loc="lower right")

        fig.suptitle("Best vs Worst Hospital AUC by Method", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / "fairness_best_worst_auc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out_dir / 'fairness_best_worst_auc.png'}")


if __name__ == "__main__":
    main()
