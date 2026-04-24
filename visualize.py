"""
Publication-ready plots for the FL benchmark:
  1. Convergence curves (AUC vs round) — one panel per α
  2. Final AUC heatmap  (algorithm × α)
  3. Test AUC vs heterogeneity α
  4. Sensitivity / specificity heatmaps
  5. Best-round heatmap
  6. Calibration heatmap (Brier score)
  7. Privacy-utility tradeoff  (ε vs AUC for DP-FedAvg)
  8. Client data distribution   (Dirichlet pie charts)
"""

from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA, FED, MODEL
from dataset import CXRBinaryDataset, base_transforms, build_image_index
from models import build_vit


COLORS = {
    "FedAvg": "#4C72B0",
    "FedProx": "#DD8452",
    "SCAFFOLD": "#55A868",
    "FedNova": "#C44E52",
    "DP-FedAvg": "#8172B2",
}


def _fl_results(all_results: list):
    return [r for r in all_results if r["algo"] not in ("Central", "Local-Only")]


def _ordered_algos_alphas(results: list):
    algos = []
    alphas = []
    for r in results:
        if r["algo"] not in algos:
            algos.append(r["algo"])
        if r["alpha"] not in alphas:
            alphas.append(r["alpha"])
    return algos, sorted(alphas)


def _metric_matrix(results: list, algos: list, alphas: list, getter):
    matrix = np.full((len(algos), len(alphas)), np.nan, dtype=float)
    for r in results:
        i = algos.index(r["algo"])
        j = alphas.index(r["alpha"])
        matrix[i, j] = getter(r)
    return matrix


def plot_convergence(all_results: list, out_dir: Path):
    fl_results = _fl_results(all_results)
    central_res = next((r for r in all_results if r["algo"] == "Central"), None)
    local_results = [r for r in all_results if r["algo"] == "Local-Only"]

    alphas = sorted(set(r["alpha"] for r in fl_results))
    fig, axes = plt.subplots(
        1,
        len(alphas),
        figsize=(5 * len(alphas), 4),
        sharey=True,
    )
    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        for r in fl_results:
            if r["alpha"] != alpha:
                continue
            rounds = [h["round"] for h in r["history"]]
            aucs = [h["auc"] for h in r["history"]]
            ax.plot(
                rounds,
                aucs,
                label=r["algo"],
                color=COLORS.get(r["algo"]),
                linewidth=2,
                marker="o",
                markersize=3,
            )

        if central_res:
            c_auc = central_res["test_metrics"]["auc"]
            ax.axhline(
                y=c_auc,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Central ({c_auc:.3f})",
            )

        local_res = next((r for r in local_results if r["alpha"] == alpha), None)
        if local_res:
            l_auc = local_res["test_metrics"]["auc"]
            ax.axhline(
                y=l_auc,
                color="gray",
                linestyle=":",
                linewidth=1.5,
                label=f"Local-Only ({l_auc:.3f})",
            )

        ax.set_title(f"α = {alpha}", fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("Validation AUC")
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(
        "FL Convergence — NIH Chest X-ray (Normal vs Suspicious)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    _save(fig, out_dir / "convergence_curves.png")


def plot_auc_heatmap(all_results: list, out_dir: Path):
    fl_results = _fl_results(all_results)
    central_res = next((r for r in all_results if r["algo"] == "Central"), None)
    local_results = [r for r in all_results if r["algo"] == "Local-Only"]

    algos, alphas = _ordered_algos_alphas(fl_results)
    matrix = _metric_matrix(
        fl_results, algos, alphas, lambda r: r["test_metrics"]["auc"]
    )

    row_labels = algos.copy()
    if central_res:
        central_row = np.full((1, len(alphas)), central_res["test_metrics"]["auc"])
        matrix = np.vstack([central_row, matrix])
        row_labels = ["── Central ──"] + row_labels
    if local_results:
        local_row = np.array([[
            next(
                (
                    r["test_metrics"]["auc"]
                    for r in local_results
                    if r["alpha"] == alpha
                ),
                np.nan,
            )
            for alpha in alphas
        ]])
        matrix = np.vstack([matrix, local_row])
        row_labels = row_labels + ["── Local-Only ──"]

    fig, ax = plt.subplots(figsize=(7, len(row_labels) * 0.7 + 1.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=[f"α={a}" for a in alphas],
        yticklabels=row_labels,
        vmin=0.5,
        vmax=1.0,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 10},
    )
    ax.set_title("Test AUC — FL Methods vs Baselines", fontsize=12)
    ax.set_xlabel("Dirichlet α  (lower = more non-IID)")
    ax.set_ylabel("Method")
    plt.tight_layout()
    _save(fig, out_dir / "auc_heatmap.png")


def plot_test_auc_vs_alpha(all_results: list, out_dir: Path):
    fl_results = _fl_results(all_results)
    central_res = next((r for r in all_results if r["algo"] == "Central"), None)
    local_results = [r for r in all_results if r["algo"] == "Local-Only"]
    algos, alphas = _ordered_algos_alphas(fl_results)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for algo in algos:
        xs = []
        ys = []
        for alpha in alphas:
            match = next(
                (
                    r for r in fl_results
                    if r["algo"] == algo and r["alpha"] == alpha
                ),
                None,
            )
            if match is None:
                continue
            xs.append(alpha)
            ys.append(match["test_metrics"]["auc"])

        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=COLORS.get(algo),
            label=algo,
        )

    if local_results:
        xs = []
        ys = []
        for alpha in alphas:
            match = next(
                (r for r in local_results if r["alpha"] == alpha),
                None,
            )
            if match is None:
                continue
            xs.append(alpha)
            ys.append(match["test_metrics"]["auc"])
        if xs:
            ax.plot(
                xs,
                ys,
                marker="s",
                linewidth=2,
                markersize=5,
                color="gray",
                linestyle="--",
                label="Local-Only",
            )

    if central_res:
        ax.axhline(
            y=central_res["test_metrics"]["auc"],
            color="black",
            linestyle=":",
            linewidth=1.8,
            label=f"Central ({central_res['test_metrics']['auc']:.3f})",
        )

    ax.set_xscale("log")
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel("Dirichlet α  (lower = more non-IID)")
    ax.set_ylabel("Test AUC")
    ax.set_title("Test AUC vs Heterogeneity")
    ax.set_ylim(0.6, 0.8)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir / "test_auc_vs_alpha.png")


def plot_clinical_metrics_heatmaps(all_results: list, out_dir: Path):
    fl_results = _fl_results(all_results)
    algos, alphas = _ordered_algos_alphas(fl_results)

    sens = _metric_matrix(
        fl_results, algos, alphas,
        lambda r: r["test_metrics"]["sensitivity"],
    )
    spec = _metric_matrix(
        fl_results, algos, alphas,
        lambda r: r["test_metrics"]["specificity"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, len(algos) * 0.8 + 1.5))
    for ax, matrix, title in zip(
        axes,
        [sens, spec],
        ["Sensitivity", "Specificity"],
    ):
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd" if title == "Sensitivity" else "Blues",
            xticklabels=[f"α={a}" for a in alphas],
            yticklabels=algos,
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 9},
        )
        ax.set_title(title)
        ax.set_xlabel("Dirichlet α")
        ax.set_ylabel("Method")

    fig.suptitle("Clinical Tradeoffs Across FL Methods", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, out_dir / "clinical_metrics_heatmaps.png")


def plot_best_round_heatmap(all_results: list, out_dir: Path):
    fl_results = _fl_results(all_results)
    algos, alphas = _ordered_algos_alphas(fl_results)
    matrix = _metric_matrix(
        fl_results, algos, alphas, lambda r: r.get("best_round", np.nan)
    )

    fig, ax = plt.subplots(figsize=(7, len(algos) * 0.7 + 1.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="crest",
        xticklabels=[f"α={a}" for a in alphas],
        yticklabels=algos,
        vmin=1,
        vmax=np.nanmax(matrix) if np.isfinite(matrix).any() else 1,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 10},
    )
    ax.set_title("Best Validation Round")
    ax.set_xlabel("Dirichlet α")
    ax.set_ylabel("Method")
    plt.tight_layout()
    _save(fig, out_dir / "best_round_heatmap.png")


def plot_brier_heatmap(all_results: list, out_dir: Path):
    fl_results = _fl_results(all_results)
    algos, alphas = _ordered_algos_alphas(fl_results)
    matrix = _metric_matrix(
        fl_results, algos, alphas, lambda r: r["test_metrics"].get("brier", np.nan)
    )

    if np.isnan(matrix).all():
        return

    fig, ax = plt.subplots(figsize=(7, len(algos) * 0.7 + 1.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="mako_r",
        xticklabels=[f"α={a}" for a in alphas],
        yticklabels=algos,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 10},
    )
    ax.set_title("Test Brier Score  (lower = better calibration)")
    ax.set_xlabel("Dirichlet α")
    ax.set_ylabel("Method")
    plt.tight_layout()
    _save(fig, out_dir / "brier_heatmap.png")


@torch.no_grad()
def _predict_probs(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        y = labels.cpu().numpy().reshape(-1).astype(int)
        all_probs.append(probs)
        all_labels.append(y)
    return np.concatenate(all_labels), np.concatenate(all_probs)


def plot_test_roc_curves(
    all_results: list,
    out_dir: Path,
    checkpoint_dir: str = "checkpoints",
    device: str | None = None,
    num_workers: int | None = None,
    splits_path: str | None = None,
):
    fl_results = _fl_results(all_results)
    if not fl_results:
        return

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    image_index = build_image_index(DATA.img_dir)
    algos, alphas = _ordered_algos_alphas(fl_results)
    num_workers = DATA.num_workers if num_workers is None else num_workers
    splits_path = Path(splits_path) if splits_path is not None else out_dir / "data_splits.pt"
    split_store = torch.load(splits_path, map_location="cpu", weights_only=False)

    fig, axes = plt.subplots(
        1, len(alphas), figsize=(5 * len(alphas), 4), sharex=True, sharey=True
    )
    if len(alphas) == 1:
        axes = [axes]

    loader_cache = {}
    model_cache = {}

    for ax, alpha in zip(axes, alphas):
        alpha_results = [r for r in fl_results if r["alpha"] == alpha]
        for r in alpha_results:
            algo = r["algo"]
            ckpt_path = Path(checkpoint_dir) / algo / f"alpha{alpha}_best.pt"
            if not ckpt_path.exists():
                continue

            if alpha not in loader_cache:
                key = next(
                    (
                        item for item in fl_results
                        if item["alpha"] == alpha
                    ),
                    None,
                )
                if key is None:
                    continue
                split_key = f"{key['algo']}_alpha{alpha}"
                split = split_store[split_key]
                test_ds = CXRBinaryDataset(
                    split["test_df"],
                    DATA.img_dir,
                    transform=base_transforms(DATA.img_size, is_train=False),
                    image_index=image_index,
                )
                loader_cache[alpha] = DataLoader(
                    test_ds,
                    batch_size=FED.batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )

            model = model_cache.get(algo)
            if model is None:
                model = build_vit(MODEL.num_classes, pretrained=False).to(device)
                model_cache[algo] = model

            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state)
            labels, probs = _predict_probs(model, loader_cache[alpha], device)
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                linewidth=2,
                color=COLORS.get(algo),
                label=f"{algo} ({roc_auc:.3f})",
            )

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_title(f"α = {alpha}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Test ROC Curves from Best Validation Checkpoints", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, out_dir / "roc_curves.png")


def plot_data_split_overview(all_results: list, out_dir: Path):
    fl_results = _fl_results(all_results)
    if not fl_results:
        return

    out_dir = Path(out_dir)
    splits_path = out_dir / "data_splits.pt"
    if not splits_path.exists():
        return

    split_store = torch.load(splits_path, map_location="cpu", weights_only=False)
    _, alphas = _ordered_algos_alphas(fl_results)
    fig, axes = plt.subplots(
        2, len(alphas), figsize=(4.5 * len(alphas), 7), sharex="col"
    )
    if len(alphas) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, alpha in enumerate(alphas):
        sample_result = next((r for r in fl_results if r["alpha"] == alpha), None)
        if sample_result is None:
            continue
        split_key = f"{sample_result['algo']}_alpha{alpha}"
        split = split_store[split_key]
        partitions = split["partitions"]

        client_ids = [f"C{i}" for i in range(len(partitions))]
        neg_counts = []
        pos_counts = []
        pos_rates = []
        for df in partitions:
            pos = (df["Finding Labels"].str.strip() != "No Finding").sum()
            neg = len(df) - pos
            neg_counts.append(neg)
            pos_counts.append(pos)
            pos_rates.append(pos / max(len(df), 1))

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        ax_top.bar(client_ids, neg_counts, color="#4C72B0", label="Normal")
        ax_top.bar(client_ids, pos_counts, bottom=neg_counts, color="#DD8452", label="Suspicious")
        ax_top.set_title(f"α = {alpha}")
        ax_top.set_ylabel("Samples")
        ax_top.grid(True, axis="y", alpha=0.25)
        if col == 0:
            ax_top.legend(fontsize=8)

        ax_bot.bar(client_ids, pos_rates, color="#55A868")
        ax_bot.set_ylim(0, 1)
        ax_bot.set_ylabel("Suspicious fraction")
        ax_bot.set_xlabel("Client")
        ax_bot.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Saved Federated Data Splits by Heterogeneity Level", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, out_dir / "data_split_overview.png")


def plot_privacy_utility(all_results: list, out_dir: Path):
    dp_results = [r for r in all_results if r["algo"] == "DP-FedAvg"]
    if not dp_results:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    alphas = sorted(set(r["alpha"] for r in dp_results))
    cmap = plt.cm.viridis(np.linspace(0.2, 0.8, len(alphas)))

    for color, alpha in zip(cmap, alphas):
        pts = [
            (r["history"][-1].get("epsilon", np.nan), r["test_metrics"]["auc"])
            for r in dp_results
            if r["alpha"] == alpha
        ]
        eps_vals, auc_vals = zip(*pts) if pts else ([], [])
        ax.scatter(
            eps_vals,
            auc_vals,
            color=color,
            s=80,
            zorder=3,
            label=f"α={alpha}",
        )

    ax.axvline(x=8, color="red", linestyle="--", alpha=0.6, label="ε=8 target")
    ax.set_xlabel("Privacy budget ε  (lower = more private)")
    ax.set_ylabel("Test AUC")
    ax.set_title("Privacy-Utility Tradeoff — DP-FedAvg")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir / "privacy_utility_tradeoff.png")


def plot_client_distributions(partitions, out_dir: Path, alpha: float):
    n = len(partitions)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (i, df) in zip(axes, enumerate(partitions)):
        pos = (df["Finding Labels"].str.strip() != "No Finding").sum()
        neg = len(df) - pos
        ax.pie(
            [neg, pos],
            labels=["Normal", "Suspicious"],
            colors=["#4C72B0", "#DD8452"],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 8},
        )
        ax.set_title(f"Hospital {i + 1}\n(n={len(df)})", fontsize=9)
    fig.suptitle(f"Client Data Distribution  α={alpha}", fontsize=11)
    plt.tight_layout()
    _save(fig, out_dir / f"client_dist_alpha{alpha}.png")


def plot_all(all_results: list, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    plot_convergence(all_results, out_dir)
    plot_auc_heatmap(all_results, out_dir)
    plot_test_auc_vs_alpha(all_results, out_dir)
    plot_clinical_metrics_heatmaps(all_results, out_dir)
    plot_best_round_heatmap(all_results, out_dir)
    plot_brier_heatmap(all_results, out_dir)
    plot_data_split_overview(all_results, out_dir)
    plot_privacy_utility(all_results, out_dir)
    print(f"\n  Plots saved → {out_dir}/")


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {path.name}")
