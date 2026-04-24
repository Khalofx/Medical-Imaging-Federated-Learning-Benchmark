"""
Evaluation utilities for binary Normal vs Suspicious classification.
Reports: AUC, accuracy, sensitivity, specificity, F1, Brier score.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs = np.vstack(all_probs).squeeze()
    labels = np.vstack(all_labels).squeeze().astype(int)
    preds = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    return dict(
        loss=total_loss / len(loader.dataset),
        auc=roc_auc_score(labels, probs),
        accuracy=accuracy_score(labels, preds),
        sensitivity=tp / max(tp + fn, 1),
        specificity=tn / max(tn + fp, 1),
        f1=f1_score(labels, preds, zero_division=0),
        brier=brier_score_loss(labels, probs),
        n=len(labels),
    )


def metrics_report(m: dict, label: str = ""):
    w = 40
    print(f"\n  {'─' * w}")
    if label:
        print(f"  {label}")
    print(f"  {'─' * w}")
    for k, v in m.items():
        if k == "n":
            print(f"  {'Samples':<18}: {v}")
        else:
            print(f"  {k.capitalize():<18}: {v:.4f}")
    print(f"  {'─' * w}")
