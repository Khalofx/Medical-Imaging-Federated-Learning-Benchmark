"""
NIH ChestX-ray14 dataset with:
  - Binary labelling (Normal vs Suspicious)
  - Patient-level train/val/test split
  - Dirichlet-based non-IID partitioning across clients
  - Optional feature shift per client (simulate scanner differences)
"""
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import DATA, FED


def build_image_index(img_dir: str) -> dict:
    """
    The NIH dataset is split across up to 12 subfolders:
        images_001/images/, images_002/images/, ..., images_012/images/

    This scans all of them once at startup and returns a dict:
        { "00000001_000.png": Path("/home/.../images_003/images/00000001_000.png"), ... }

    The dataset class uses this for O(1) lookups instead of
    searching subfolders on every __getitem__ call.
    """
    index = {}
    base = Path(img_dir)

    subfolders = sorted(base.glob("images_*/images"))
    if not subfolders:
        subfolders = [base / "images"]

    for folder in subfolders:
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                index[f.name] = f

    print(
        f"  Image index built: {len(index):,} files found "
        f"across {len(subfolders)} subfolder(s)."
    )
    if len(index) == 0:
        raise FileNotFoundError(
            f"No images found under {img_dir}. "
            "Check that your path contains images_001/, images_002/, ... subfolders."
        )
    return index


# ─────────────────────────────────────────────────────────────
# 1.  Transforms
# ─────────────────────────────────────────────────────────────
def base_transforms(img_size: int, is_train: bool,
                    gamma: float = 1.0,        # feature-shift: brightness
                    blur : bool  = False,      # feature-shift: scanner blur
) -> transforms.Compose:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ops = []
    if is_train:
        ops += [
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        ]
    else:
        ops += [transforms.Resize((img_size, img_size))]

    if blur:
        ops.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if gamma != 1.0:
        # simulate scanner brightness difference
        ops.append(transforms.Lambda(lambda x: x * gamma))
    return transforms.Compose(ops)


# ─────────────────────────────────────────────────────────────
# 2.  Dataset class
# ─────────────────────────────────────────────────────────────
class CXRBinaryDataset(Dataset):
    """Binary Normal (0) vs Suspicious (1) chest X-ray dataset."""

    def __init__(self, df: pd.DataFrame, img_dir: str,
                 transform=None,
                 image_index: dict = None):
        self.df          = df.reset_index(drop=True)
        self.transform   = transform
        # Accept a pre-built index or build one now (slower — prefer passing it in)
        self.image_index = image_index or build_image_index(img_dir)
        self._warned_bad_files = set()

    def __len__(self):
        return len(self.df)

    def _warn_bad_file(self, filename: str, reason: str):
        key = (filename, reason)
        if key not in self._warned_bad_files:
            warnings.warn(
                f"Skipping unreadable image '{filename}': {reason}",
                RuntimeWarning,
            )
            self._warned_bad_files.add(key)

    def __getitem__(self, idx):
        for offset in range(len(self.df)):
            row = self.df.iloc[(idx + offset) % len(self.df)]
            filename = row["Image Index"]
            img_path = self.image_index.get(filename)

            if img_path is None:
                self._warn_bad_file(filename, "file not found in image index")
                continue

            try:
                with Image.open(img_path) as pil_img:
                    img = pil_img.convert("RGB")
            except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError) as exc:
                self._warn_bad_file(filename, str(exc))
                continue

            if self.transform:
                img = self.transform(img)
            label = torch.tensor(
                [0.0] if row["Finding Labels"].strip() == "No Finding" else [1.0],
                dtype=torch.float32,
            )
            return img, label

        raise RuntimeError("No readable images were found in this dataset partition.")


# ─────────────────────────────────────────────────────────────
# 3.  Patient-level split
# ─────────────────────────────────────────────────────────────
def patient_level_split(csv_path: str,
                         seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split at patient level (not image level) to prevent data leakage.
    Returns (train_df, val_df, test_df) — 70/15/15.
    """
    df       = pd.read_csv(csv_path)[["Image Index", "Finding Labels", "Patient ID"]]
    patients = df["Patient ID"].unique()
    tr_p, tmp_p = train_test_split(patients, test_size=0.30, random_state=seed)
    vl_p, ts_p  = train_test_split(tmp_p,    test_size=0.50, random_state=seed)
    return (df[df["Patient ID"].isin(tr_p)],
            df[df["Patient ID"].isin(vl_p)],
            df[df["Patient ID"].isin(ts_p)])


# ─────────────────────────────────────────────────────────────
# 4.  Dirichlet non-IID partitioning
# ─────────────────────────────────────────────────────────────
def dirichlet_partition(train_df    : pd.DataFrame,
                         num_clients : int,
                         alpha       : float,
                         seed        : int = 42,
) -> List[pd.DataFrame]:
    """
    Partition training data across clients using a Dirichlet distribution
    over the binary label (Normal=0, Suspicious=1).

    alpha → 0   : each client sees almost only one class (maximally non-IID)
    alpha → ∞   : each client has the global class distribution (IID)

    Returns a list of per-client DataFrames.
    """
    rng    = np.random.default_rng(seed)
    labels = (train_df["Finding Labels"].str.strip() != "No Finding").astype(int).values
    idxs   = np.arange(len(train_df))

    # Sample proportion vectors per class from Dirichlet
    class_indices = [idxs[labels == c] for c in [0, 1]]
    client_idx: List[List[int]] = [[] for _ in range(num_clients)]

    for cls_idx in class_indices:
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        # Ensure no client gets zero samples
        proportions = np.maximum(proportions, 1e-3)
        proportions /= proportions.sum()
        cuts = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        for c, chunk in enumerate(np.split(cls_idx, cuts)):
            client_idx[c].extend(chunk.tolist())

    partitions = []
    for c in range(num_clients):
        rng.shuffle(client_idx[c])
        partitions.append(train_df.iloc[client_idx[c]])

    _print_partition_stats(partitions, alpha)
    return partitions


def _print_partition_stats(partitions: List[pd.DataFrame], alpha: float):
    print(f"\n  Dirichlet partition  α={alpha}")
    print(f"  {'Client':<10} {'N':>6}  {'%Suspicious':>12}")
    for i, df in enumerate(partitions):
        n    = len(df)
        pos  = (df["Finding Labels"].str.strip() != "No Finding").sum()
        print(f"  Client {i:<4} {n:>6}  {pos/n*100:>11.1f}%")


# ─────────────────────────────────────────────────────────────
# 5.  Feature-shift simulation (optional scanner heterogeneity)
# ─────────────────────────────────────────────────────────────
def scanner_profiles(num_clients: int, seed: int = 42) -> List[Dict]:
    """
    Assign each client a random scanner profile:
      - gamma : brightness multiplier  (0.8 – 1.2)
      - blur  : Gaussian blur toggle
    Simulates different X-ray machine manufacturers across hospitals.
    """
    rng = np.random.default_rng(seed)
    profiles = []
    for _ in range(num_clients):
        profiles.append(dict(
            gamma=float(rng.uniform(0.85, 1.15)),
            blur =bool(rng.choice([True, False], p=[0.3, 0.7])),
        ))
    return profiles


# ─────────────────────────────────────────────────────────────
# 6.  DataLoader factory
# ─────────────────────────────────────────────────────────────
def get_pos_weight(df: pd.DataFrame) -> torch.Tensor:
    pos = (df["Finding Labels"].str.strip() != "No Finding").sum()
    neg = len(df) - pos
    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32)


def make_loaders(client_df   : pd.DataFrame,
                 val_df      : pd.DataFrame,
                 img_dir     : str,
                 img_size    : int,
                 batch_size  : int,
                 num_workers : int,
                 scanner     : Dict = None,
                 image_index : dict = None,
) -> Tuple[DataLoader, DataLoader]:
    sc  = scanner or {"gamma": 1.0, "blur": False}
    idx = image_index or build_image_index(img_dir)
    train_ds = CXRBinaryDataset(
        client_df, img_dir,
        transform=base_transforms(img_size, is_train=True, **sc),
        image_index=idx,
    )
    val_ds = CXRBinaryDataset(
        val_df, img_dir,
        transform=base_transforms(img_size, is_train=False),
        image_index=idx,
    )
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw))
