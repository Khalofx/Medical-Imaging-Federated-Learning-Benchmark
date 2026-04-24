# config.py
"""
Central configuration for the FL Chest X-ray Benchmark.
All experiments are driven by this file — edit here, not in algorithm files.
"""
import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    data_dir   : str = "/home/khalifa.juma/HC701/Project/NIH_Chest_Xrays"
    img_dir    : str = "/home/khalifa.juma/HC701/Project/NIH_Chest_Xrays"
    csv_path   : str = "/home/khalifa.juma/HC701/Project/NIH_Chest_Xrays/Data_Entry_2017.csv"
    img_size   : int = 224
    num_workers: int = int(os.environ.get("NUM_WORKERS", "4"))
    seed       : int = 42
    # Dirichlet heterogeneity levels to sweep
    # α → 0 = maximally non-IID (siloed), α → ∞ = IID
    alpha_values: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 5.0])

# Detect smoke test mode — set by run_smoke_test.sh
_SMOKE = os.environ.get("FL_SMOKE_TEST", "0") == "1"

@dataclass
class FederatedConfig:
    num_clients  : int   = 5
    num_rounds   : int   = 2  if _SMOKE else 20   # 2 rounds for smoke, 20 for full
    local_epochs : int   = 1  if _SMOKE else 3    # 1 epoch  for smoke, 3  for full
    batch_size   : int   = 32
    fraction_fit : float = 1.0

@dataclass
class ModelConfig:
    architecture: str  = "vit_base_patch16_224"
    pretrained  : bool = True
    num_classes : int  = 1          # binary: Normal vs Suspicious

@dataclass
class TrainConfig:
    lr            : float = 1e-4
    weight_decay  : float = 1e-2
    # FedProx
    mu            : float = 0.01    # proximal term strength
    # SCAFFOLD  — no extra HP beyond lr
    # FedNova
    rho           : float = 0.0     # momentum for local normalisation
    # Ditto
    lambda_ditto  : float = 0.1     # personalisation strength

@dataclass
class PrivacyConfig:
    # Differential Privacy (DP-SGD via Opacus)
    dp_enabled        : bool  = True
    dp_batch_size     : int   = 4     # smaller batch for Opacus per-sample grads
    target_epsilon    : float = 8.0   # privacy budget (lower = more private)
    target_delta      : float = 1e-5  # failure probability
    max_grad_norm     : float = 1.0   # clipping threshold
    noise_multiplier  : float = 1.1   # Gaussian noise σ; set None to auto-compute

    # Secure Aggregation (simulated — masks cancel during sum)
    secagg_enabled    : bool  = True
    secagg_bits       : int   = 16    # precision bits for quantised masks

    # Gradient clipping (always on even without full DP)
    clip_norm         : float = 1.0

@dataclass
class BenchmarkConfig:
    algorithms    : List[str] = field(default_factory=lambda:
                        ["FedAvg", "FedProx", "SCAFFOLD", "FedNova", "DP-FedAvg"])
    results_dir   : str  = "./results"
    checkpoint_dir: str  = "./checkpoints"
    device        : str  = "cuda"   # overridden at runtime if no GPU
    # DP-FedAvg is currently skipped at runtime in benchmark.py because the
    # ViT + Opacus stack is not compatible in this environment.

# ── Convenience singleton ──────────────────────────────────────────────────
DATA    = DataConfig()
FED     = FederatedConfig()
MODEL   = ModelConfig()
TRAIN   = TrainConfig()
PRIVACY = PrivacyConfig()
BENCH   = BenchmarkConfig()
