# ══════════════════════════════════════════════════════════════
# evaluate.py
# ══════════════════════════════════════════════════════════════
"""
Evaluation utilities for binary Normal vs Suspicious classification.
Reports: AUC, accuracy, sensitivity, specificity, F1, Brier score.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, brier_score_loss,
)


@torch.no_grad()
def evaluate_model(model     : nn.Module,
                   loader    : DataLoader,
                   criterion : nn.Module,
                   device    : str) -> dict:
    model.eval()
    total_loss  = 0.0
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs  = np.vstack(all_probs).squeeze()
    labels = np.vstack(all_labels).squeeze().astype(int)
    preds  = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()

    return dict(
        loss        = total_loss / len(loader.dataset),
        auc         = roc_auc_score(labels, probs),
        accuracy    = accuracy_score(labels, preds),
        sensitivity = tp / max(tp + fn, 1),    # recall for Suspicious
        specificity = tn / max(tn + fp, 1),    # recall for Normal
        f1          = f1_score(labels, preds, zero_division=0),
        brier       = brier_score_loss(labels, probs),
        n           = len(labels),
    )


def metrics_report(m: dict, label: str = ""):
    w = 40
    print(f"\n  {'─'*w}")
    if label:
        print(f"  {label}")
    print(f"  {'─'*w}")
    for k, v in m.items():
        if k == "n":
            print(f"  {'Samples':<18}: {v}")
        else:
            print(f"  {k.capitalize():<18}: {v:.4f}")
    print(f"  {'─'*w}")


# ══════════════════════════════════════════════════════════════
# visualize.py
# ══════════════════════════════════════════════════════════════
"""
Publication-ready plots for the FL benchmark:
  1. Convergence curves (AUC vs round) — one panel per α
  2. Final AUC heatmap  (algorithm × α)
  3. Privacy-utility tradeoff  (ε vs AUC for DP-FedAvg)
  4. Client data distribution   (Dirichlet pie charts)
"""
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

COLORS = {
    "FedAvg"   : "#4C72B0",
    "FedProx"  : "#DD8452",
    "SCAFFOLD" : "#55A868",
    "FedNova"  : "#C44E52",
    "DP-FedAvg": "#8172B2",
}


# ─── 1.  Convergence curves ───────────────────────────────────
def plot_convergence(all_results: list, out_dir: Path):
    fl_results   = [r for r in all_results
                    if r["algo"] not in ("Central", "Local-Only")]
    central_res  = next((r for r in all_results if r["algo"] == "Central"), None)
    local_res    = next((r for r in all_results if r["algo"] == "Local-Only"), None)

    alphas = sorted(set(r["alpha"] for r in fl_results))
    fig, axes = plt.subplots(1, len(alphas),
                              figsize=(5 * len(alphas), 4),
                              sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        # FL algorithm curves
        for r in fl_results:
            if r["alpha"] != alpha:
                continue
            rounds = [h["round"] for h in r["history"]]
            aucs   = [h["auc"]   for h in r["history"]]
            ax.plot(rounds, aucs,
                    label=r["algo"],
                    color=COLORS.get(r["algo"]),
                    linewidth=2, marker="o", markersize=3)

        # Central baseline — horizontal dashed line (ceiling)
        if central_res:
            c_auc = central_res["test_metrics"]["auc"]
            ax.axhline(y=c_auc, color="black", linestyle="--",
                       linewidth=1.5, label=f"Central ({c_auc:.3f})")

        # Local-only baseline — horizontal dotted line (floor)
        if local_res:
            l_auc = local_res["test_metrics"]["auc"]
            ax.axhline(y=l_auc, color="gray", linestyle=":",
                       linewidth=1.5, label=f"Local-Only ({l_auc:.3f})")

        ax.set_title(f"α = {alpha}", fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("Validation AUC")
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("FL Convergence — NIH Chest X-ray (Normal vs Suspicious)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, out_dir / "convergence_curves.png")


# ─── 2.  AUC heatmap ──────────────────────────────────────────
def plot_auc_heatmap(all_results: list, out_dir: Path):
    fl_results  = [r for r in all_results
                   if r["algo"] not in ("Central", "Local-Only")]
    central_res = next((r for r in all_results if r["algo"] == "Central"), None)
    local_res   = next((r for r in all_results if r["algo"] == "Local-Only"), None)

    algos  = []
    alphas = []
    for r in fl_results:
        if r["algo"]  not in algos:  algos.append(r["algo"])
        if r["alpha"] not in alphas: alphas.append(r["alpha"])
    alphas = sorted(alphas)

    matrix = np.zeros((len(algos), len(alphas)))
    for r in fl_results:
        i = algos.index(r["algo"])
        j = alphas.index(r["alpha"])
        matrix[i, j] = r["test_metrics"]["auc"]

    # Add baseline rows
    row_labels = algos.copy()
    if central_res:
        central_row = np.full((1, len(alphas)),
                               central_res["test_metrics"]["auc"])
        matrix     = np.vstack([central_row, matrix])
        row_labels = ["── Central ──"] + row_labels
    if local_res:
        local_row  = np.full((1, len(alphas)),
                              local_res["test_metrics"]["auc"])
        matrix     = np.vstack([matrix, local_row])
        row_labels = row_labels + ["── Local-Only ──"]

    fig, ax = plt.subplots(figsize=(7, len(row_labels) * 0.7 + 1.5))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=[f"α={a}" for a in alphas],
                yticklabels=row_labels,
                vmin=0.5, vmax=1.0, ax=ax,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 10})
    ax.set_title("Test AUC — FL Methods vs Baselines", fontsize=12)
    ax.set_xlabel("Dirichlet α  (lower = more non-IID)")
    ax.set_ylabel("Method")
    plt.tight_layout()
    _save(fig, out_dir / "auc_heatmap.png")


# ─── 3.  Privacy-utility tradeoff ─────────────────────────────
def plot_privacy_utility(all_results: list, out_dir: Path):
    dp_results = [r for r in all_results if r["algo"] == "DP-FedAvg"]
    if not dp_results:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    alphas  = sorted(set(r["alpha"] for r in dp_results))
    cmap    = plt.cm.viridis(np.linspace(0.2, 0.8, len(alphas)))

    for color, alpha in zip(cmap, alphas):
        pts = [(r["history"][-1].get("epsilon", np.nan),
                r["test_metrics"]["auc"])
               for r in dp_results if r["alpha"] == alpha]
        eps_vals, auc_vals = zip(*pts) if pts else ([], [])
        ax.scatter(eps_vals, auc_vals, color=color, s=80, zorder=3,
                   label=f"α={alpha}")

    ax.axvline(x=8, color="red", linestyle="--", alpha=0.6,
               label="ε=8 target")
    ax.set_xlabel("Privacy budget ε  (lower = more private)")
    ax.set_ylabel("Test AUC")
    ax.set_title("Privacy-Utility Tradeoff — DP-FedAvg")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir / "privacy_utility_tradeoff.png")


# ─── 4.  Per-client label distribution ───────────────────────
def plot_client_distributions(partitions, out_dir: Path, alpha: float):
    n = len(partitions)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (i, df) in zip(axes, enumerate(partitions)):
        pos = (df["Finding Labels"].str.strip() != "No Finding").sum()
        neg = len(df) - pos
        ax.pie([neg, pos],
               labels=["Normal", "Suspicious"],
               colors=["#4C72B0", "#DD8452"],
               autopct="%1.0f%%", startangle=90,
               textprops={"fontsize": 8})
        ax.set_title(f"Hospital {i+1}\n(n={len(df)})", fontsize=9)
    fig.suptitle(f"Client Data Distribution  α={alpha}", fontsize=11)
    plt.tight_layout()
    _save(fig, out_dir / f"client_dist_alpha{alpha}.png")


# ─── Master caller ─────────────────────────────────────────────
def plot_all(all_results: list, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    plot_convergence(all_results, out_dir)
    plot_auc_heatmap(all_results, out_dir)
    plot_privacy_utility(all_results, out_dir)
    print(f"\n  Plots saved → {out_dir}/")


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {path.name}")
