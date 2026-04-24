"""
=============================================================
  Federated Vision Transformer — NIH Chest X-ray14
  Replication of: NIH-Chest-X-rays-Multi-Label-Image-Classification
  Framework : PyTorch + Flower (flwr) for Federated Learning
=============================================================

INSTALL DEPENDENCIES:
    pip install torch torchvision timm flwr scikit-learn \
                pandas numpy matplotlib tqdm kaggle

DATASET DOWNLOAD (NIH ChestX-ray14):
    Option A – Kaggle CLI (recommended, ~45 GB):
        1. Get your Kaggle API key → kaggle.com → Account → Create API Token
        2. Place kaggle.json in ~/.kaggle/
        3. Run:
           kaggle datasets download -d nih-chest-xrays/data
           unzip data.zip -d ./nih_chest_xray

    Option B – Direct NIH link:
        https://nihcc.app.box.com/v/ChestXray-NIHCC
        Download all images_*.tar.gz and Data_Entry_2017.csv

EXPECTED DIRECTORY LAYOUT:
    ./nih_chest_xray/
        Data_Entry_2017.csv
        images/
            00000001_000.png
            00000001_001.png
            ...
"""

# ─────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────
import os, random, copy
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

try:
    import flwr as fl                # Federated Learning
except ImportError:
    fl = None

# ─────────────────────────────────────────────
# 1.  Configuration
# ─────────────────────────────────────────────
CFG = dict(
    data_dir        = "./nih_chest_xray",
    img_dir         = "./nih_chest_xray/images",
    csv_path        = "./nih_chest_xray/Data_Entry_2017.csv",
    img_size        = 224,
    batch_size      = 32,
    num_epochs      = 25,           # local epochs per FL round
    num_rounds      = 5,            # federated rounds
    num_clients     = 5,            # simulated hospital nodes
    lr              = 1e-4,
    weight_decay    = 1e-2,
    seed            = 42,
    device          = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers     = 4,
)

DISEASES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural_Thickening","Hernia",
]
NUM_CLASSES = 1          # Binary: Normal (0) vs Suspicious (1)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])

# ─────────────────────────────────────────────
# 2.  Dataset
# ─────────────────────────────────────────────
class ChestXrayDataset(Dataset):
    """Multi-label NIH ChestX-ray14 dataset."""

    def __init__(self, df: pd.DataFrame, img_dir: str,
                 transform=None, is_train=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["Image Index"]

        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Binary label: 0 = Normal, 1 = Suspicious (any finding present)
        labels_str = row["Finding Labels"]
        label = torch.tensor(
            [0.0] if labels_str.strip() == "No Finding" else [1.0],
            dtype=torch.float32,
        )
        return img, label


def get_transforms(img_size: int, is_train: bool):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_dataframes(cfg: dict):
    """
    Patient-level split → train 70 / val 15 / test 15.
    Returns three DataFrames.
    """
    df = pd.read_csv(cfg["csv_path"])
    df = df[["Image Index", "Finding Labels", "Patient ID"]]

    patients = df["Patient ID"].unique()
    train_p, temp_p = train_test_split(patients, test_size=0.30,
                                       random_state=cfg["seed"])
    val_p, test_p   = train_test_split(temp_p,   test_size=0.50,
                                       random_state=cfg["seed"])

    train_df = df[df["Patient ID"].isin(train_p)]
    val_df   = df[df["Patient ID"].isin(val_p)]
    test_df  = df[df["Patient ID"].isin(test_p)]
    return train_df, val_df, test_df


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Positive-frequency weight for binary Normal vs Suspicious.
    pos_weight = num_negatives / num_positives  (passed to BCEWithLogitsLoss).
    """
    n = len(df)
    pos = (~df["Finding Labels"].str.strip().eq("No Finding")).sum()
    neg = n - pos
    print(f"  Class balance — Normal: {neg} ({neg/n*100:.1f}%)  "
          f"Suspicious: {pos} ({pos/n*100:.1f}%)")
    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32)


def get_binary_metrics(probs: np.ndarray, labels: np.ndarray):
    """Accuracy, AUC, sensitivity, specificity for binary output."""
    from sklearn.metrics import (roc_auc_score, confusion_matrix,
                                 accuracy_score)
    preds = (probs >= 0.5).astype(int)
    auc   = roc_auc_score(labels, probs)
    acc   = accuracy_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / max(tp + fn, 1)   # recall for Suspicious
    specificity = tn / max(tn + fp, 1)   # recall for Normal
    return dict(auc=auc, accuracy=acc,
                sensitivity=sensitivity, specificity=specificity)


def partition_for_federated(train_df: pd.DataFrame,
                             num_clients: int) -> List[pd.DataFrame]:
    """
    Split training data into `num_clients` non-IID partitions
    (grouped by patient to simulate different hospital populations).
    """
    patients = train_df["Patient ID"].unique()
    np.random.shuffle(patients)
    splits = np.array_split(patients, num_clients)
    return [train_df[train_df["Patient ID"].isin(s)] for s in splits]


# ─────────────────────────────────────────────
# 3.  Model — Vision Transformer (ViT-Base/16)
# ─────────────────────────────────────────────
def build_vit(num_classes: int = NUM_CLASSES,
              pretrained: bool = True) -> nn.Module:
    """
    Build a ViT classifier for the binary chest X-ray task.
    Prefer `timm` when available and fall back to torchvision so
    the benchmark can still run in leaner environments.
    """
    try:
        import timm

        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        return model
    except ImportError:
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = vit_b_16(weights=weights)
        except Exception:
            model = vit_b_16(weights=None)

        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model


# ─────────────────────────────────────────────
# 4.  Training / Evaluation Utilities
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(loader, leave=False, desc="  train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for imgs, labels in tqdm(loader, leave=False, desc="  eval "):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.vstack(all_probs).squeeze()    # shape (N,)
    all_labels = np.vstack(all_labels).squeeze()   # shape (N,)

    metrics = get_binary_metrics(all_probs, all_labels)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics, all_probs, all_labels


# ─────────────────────────────────────────────
# 5.  Federated Learning — Flower Client
# ─────────────────────────────────────────────
def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model: nn.Module, params: List[np.ndarray]):
    state = model.state_dict()
    for k, v in zip(state.keys(), params):
        state[k] = torch.tensor(v)
    model.load_state_dict(state)


class CXRClient(fl.client.NumPyClient if fl is not None else object):

    def __init__(self, cid: int, model: nn.Module,
                 train_df: pd.DataFrame, val_df: pd.DataFrame,
                 cfg: dict, class_weights: torch.Tensor):
        self.cid   = cid
        self.model = model.to(cfg["device"])
        self.cfg   = cfg
        self.device = cfg["device"]

        train_ds = ChestXrayDataset(
            train_df, cfg["img_dir"],
            transform=get_transforms(cfg["img_size"], is_train=True),
            is_train=True,
        )
        val_ds = ChestXrayDataset(
            val_df, cfg["img_dir"],
            transform=get_transforms(cfg["img_size"], is_train=False),
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"],
            shuffle=True, num_workers=cfg["num_workers"], pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg["batch_size"],
            shuffle=False, num_workers=cfg["num_workers"], pin_memory=True,
        )
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=class_weights.to(self.device)
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg["num_epochs"]
        )

    # ── Flower hooks ──────────────────────────
    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        for epoch in range(self.cfg["num_epochs"]):
            loss = train_one_epoch(
                self.model, self.train_loader,
                self.optimizer, self.criterion, self.device,
            )
            self.scheduler.step()
            print(f"  [Client {self.cid}] Epoch {epoch+1}/{self.cfg['num_epochs']}  loss={loss:.4f}")
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, metrics, _, _ = evaluate(
            self.model, self.val_loader, self.criterion, self.device
        )
        print(f"  [Client {self.cid}] Val  loss={loss:.4f}  "
              f"AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']*100:.2f}%  "
              f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}")
        return float(loss), len(self.val_loader.dataset), {
            "auc": metrics["auc"], "accuracy": float(metrics["accuracy"]),
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
        }


# ─────────────────────────────────────────────
# 6.  Federated Simulation Entry-Point
# ─────────────────────────────────────────────
def run_federated(cfg: dict = CFG):
    if fl is None:
        raise ImportError("run_federated requires Flower. Install it with: pip install flwr")

    print("=" * 60)
    print("  Federated ViT — NIH Chest X-ray14")
    print("=" * 60)

    # ── Data ──────────────────────────────────
    print("\n[1/4] Loading & partitioning data...")
    train_df, val_df, test_df = load_dataframes(cfg)
    client_dfs   = partition_for_federated(train_df, cfg["num_clients"])
    class_weights = compute_class_weights(train_df)
    print(f"  Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    # ── Global model ──────────────────────────
    print("\n[2/4] Building ViT-Base/16 (pretrained=ImageNet21k)...")
    global_model = build_vit(pretrained=True)

    # ── Flower client factory ─────────────────
    def client_fn(cid: str) -> fl.client.Client:
        i = int(cid)
        local_model = copy.deepcopy(global_model)
        return CXRClient(
            cid=i,
            model=local_model,
            train_df=client_dfs[i],
            val_df=val_df,
            cfg=cfg,
            class_weights=class_weights,
        )

    # ── FedAvg strategy ───────────────────────
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=cfg["num_clients"],
        min_evaluate_clients=cfg["num_clients"],
        min_available_clients=cfg["num_clients"],
        initial_parameters=fl.common.ndarrays_to_parameters(
            get_parameters(global_model)
        ),
    )

    # ── Run simulation ────────────────────────
    print(f"\n[3/4] Starting FL simulation  "
          f"({cfg['num_clients']} clients × {cfg['num_rounds']} rounds)...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg["num_clients"],
        config=fl.server.ServerConfig(num_rounds=cfg["num_rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": 2,
                          "num_gpus": 0.25 if cfg["device"] == "cuda" else 0},
    )

    # ── Final test evaluation ─────────────────
    print("\n[4/4] Final evaluation on held-out test set...")
    # Retrieve aggregated weights from last round
    final_params = strategy.parameters  # numpy arrays
    if final_params is not None:
        set_parameters(global_model, fl.common.parameters_to_ndarrays(final_params))

    global_model = global_model.to(cfg["device"])
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=class_weights.to(cfg["device"])
    )
    test_ds = ChestXrayDataset(
        test_df, cfg["img_dir"],
        transform=get_transforms(cfg["img_size"], is_train=False),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"],
    )
    test_loss, metrics, probs, labels = evaluate(
        global_model, test_loader, criterion, cfg["device"]
    )

    print(f"\n{'─'*40}")
    print(f"  Test Loss   : {test_loss:.4f}")
    print(f"  AUC         : {metrics['auc']:.4f}")
    print(f"  Accuracy    : {metrics['accuracy']*100:.2f}%")
    print(f"  Sensitivity : {metrics['sensitivity']:.4f}  (True Suspicious rate)")
    print(f"  Specificity : {metrics['specificity']:.4f}  (True Normal rate)")
    print(f"{'─'*40}")

    # ── Save model ────────────────────────────
    out_path = Path("./checkpoints/federated_vit_final.pt")
    out_path.parent.mkdir(exist_ok=True)
    torch.save(global_model.state_dict(), out_path)
    print(f"\nModel saved → {out_path}")

    plot_history(history)
    return global_model, history


# ─────────────────────────────────────────────
# 7.  Plotting
# ─────────────────────────────────────────────
def plot_history(history):
    """Plot federated training & validation metrics."""
    rounds = range(1, len(history.losses_distributed) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    losses = [l for _, l in history.losses_distributed]
    axes[0].plot(rounds, losses, marker="o")
    axes[0].set_title("Federated Val Loss per Round")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # AUC (if available in metrics)
    if history.metrics_distributed:
        aucs = [m.get("auc", 0) for _, m in history.metrics_distributed.get("auc", [])]
        if aucs:
            axes[1].plot(range(1, len(aucs)+1), aucs, marker="o", color="green")
            axes[1].set_title("Federated Mean AUC per Round")
            axes[1].set_xlabel("Round"); axes[1].set_ylabel("AUC")
            axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("./checkpoints/fl_training_history.png", dpi=150)
    plt.show()
    print("Plot saved → ./checkpoints/fl_training_history.png")


# ─────────────────────────────────────────────
# 8.  Standalone (non-federated) Training
#     Use this first to verify everything works
# ─────────────────────────────────────────────
def run_standalone(cfg: dict = CFG):
    """
    Single-machine training for quick sanity check / baseline.
    Mirrors the paper's setup: ViT-Base, 25 epochs, AdamW, cosine LR.
    """
    print("=" * 60)
    print("  Standalone ViT — NIH Chest X-ray14  (Sanity Check)")
    print("=" * 60)

    train_df, val_df, test_df = load_dataframes(cfg)
    class_weights = compute_class_weights(train_df)

    train_ds = ChestXrayDataset(
        train_df, cfg["img_dir"],
        transform=get_transforms(cfg["img_size"], is_train=True),
        is_train=True,
    )
    val_ds  = ChestXrayDataset(val_df, cfg["img_dir"],
                               transform=get_transforms(cfg["img_size"], False))
    test_ds = ChestXrayDataset(test_df, cfg["img_dir"],
                               transform=get_transforms(cfg["img_size"], False))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"])
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"])

    device    = cfg["device"]
    model     = build_vit(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["num_epochs"])

    best_auc, best_state = 0.0, None
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, cfg["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['num_epochs']}")
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, metrics, _, _ = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(metrics["auc"])

        print(f"  tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_auc={metrics['auc']:.4f}  "
              f"sens={metrics['sensitivity']:.4f}  "
              f"spec={metrics['specificity']:.4f}")

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())
            print("  ✓ New best model saved")

    # Final test
    model.load_state_dict(best_state)
    test_loss, metrics, probs, labels = evaluate(
        model, test_loader, criterion, device)
    print(f"\n{'─'*40}")
    print(f"  Test AUC        : {metrics['auc']:.4f}")
    print(f"  Test Accuracy   : {metrics['accuracy']*100:.2f}%")
    print(f"  Sensitivity     : {metrics['sensitivity']:.4f}")
    print(f"  Specificity     : {metrics['specificity']:.4f}")

    out = Path("./checkpoints/standalone_vit_best.pt")
    out.parent.mkdir(exist_ok=True)
    torch.save(best_state, out)
    print(f"  Saved → {out}")

    return model, history


# ─────────────────────────────────────────────
# 9.  Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standalone", "federated"],
                        default="standalone",
                        help="standalone = single machine baseline | "
                             "federated = FL simulation with Flower")
    args = parser.parse_args()

    if args.mode == "standalone":
        run_standalone(CFG)
    else:
        run_federated(CFG)
