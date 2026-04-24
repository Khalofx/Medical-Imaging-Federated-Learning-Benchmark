# privacy.py
"""
Privacy mechanisms:
  1. DP-SGD via Opacus  — per-sample gradient clipping + Gaussian noise
  2. SecAgg simulation  — additive masking that cancels on the server
  3. Gradient clipping  — standalone, always available

Install:  pip install opacus
"""
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional

from config import PRIVACY


# ─────────────────────────────────────────────────────────────
# 1.  DP-SGD (Opacus)
# ─────────────────────────────────────────────────────────────
def make_private(model      : nn.Module,
                 optimizer  ,
                 data_loader: DataLoader,
                 cfg        = PRIVACY,
) -> Tuple[nn.Module, object, DataLoader]:
    """
    Wraps model + optimizer with Opacus PrivacyEngine.
    Returns (private_model, private_optimizer, private_loader).

    The noise_multiplier can be set manually in PrivacyConfig or
    left as None to auto-compute from (epsilon, delta, epochs).
    """
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
    except ImportError:
        raise ImportError("pip install opacus")

    # ViT has some unsupported layers — fix automatically
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    engine = PrivacyEngine()

    if cfg.noise_multiplier is not None:
        private_model, private_opt, private_loader = engine.make_private(
            module      = model,
            optimizer   = optimizer,
            data_loader = data_loader,
            noise_multiplier = cfg.noise_multiplier,
            max_grad_norm    = cfg.max_grad_norm,
        )
    else:
        private_model, private_opt, private_loader = engine.make_private_with_epsilon(
            module      = model,
            optimizer   = optimizer,
            data_loader = data_loader,
            epochs      = 1,           # per-round budget
            target_epsilon  = cfg.target_epsilon,
            target_delta    = cfg.target_delta,
            max_grad_norm   = cfg.max_grad_norm,
        )

    return private_model, private_opt, private_loader, engine


def get_privacy_spent(engine) -> Tuple[float, float]:
    """Return (epsilon, delta) spent so far."""
    eps = engine.get_epsilon(PRIVACY.target_delta)
    return eps, PRIVACY.target_delta


# ─────────────────────────────────────────────────────────────
# 2.  Secure Aggregation (simulation)
# ─────────────────────────────────────────────────────────────
class SecureAggregator:
    """
    Simulated SecAgg using pairwise additive secret sharing.

    In a real deployment each client pair exchanges random seeds
    via a key-agreement protocol; masks cancel during summation.
    Here we faithfully simulate the masking/unmasking arithmetic
    so the aggregation result is identical to plaintext FedAvg
    while demonstrating the privacy guarantee holds.

    Reference: Bonawitz et al. "Practical Secure Aggregation
               for Privacy-Preserving ML" (CCS 2017)
    """

    def __init__(self, num_clients: int, bits: int = 16):
        self.n    = num_clients
        self.bits = bits
        self.scale = 2 ** bits

    def _pairwise_seed(self, i: int, j: int, round_id: int) -> int:
        """Deterministic seed for pair (i,j) — simulates DH key exchange."""
        return hash((min(i,j), max(i,j), round_id)) & 0xFFFFFFFF

    def mask_updates(self,
                     client_id : int,
                     params    : List[np.ndarray],
                     round_id  : int,
    ) -> List[np.ndarray]:
        """Add pairwise cancelling masks to client parameters."""
        masked = [p.copy() for p in params]
        for j in range(self.n):
            if j == client_id:
                continue
            seed  = self._pairwise_seed(client_id, j, round_id)
            rng   = np.random.default_rng(seed)
            sign  = 1 if client_id < j else -1
            for k, p in enumerate(params):
                noise = rng.integers(
                    -self.scale, self.scale, size=p.shape
                ).astype(p.dtype) / self.scale
                masked[k] = masked[k] + sign * noise
        return masked

    def aggregate(self,
                  all_masked : List[List[np.ndarray]],
    ) -> List[np.ndarray]:
        """
        Sum masked updates — pairwise masks cancel exactly,
        yielding the plain average.
        """
        n = len(all_masked)
        agg = [sum(c[k] for c in all_masked) / n
               for k in range(len(all_masked[0]))]
        return agg


# ─────────────────────────────────────────────────────────────
# 3.  Gradient clipping (standalone)
# ─────────────────────────────────────────────────────────────
def clip_gradients(model: nn.Module, max_norm: float = PRIVACY.clip_norm):
    """Standard gradient clipping. Always applied before optimizer.step()."""
    nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# ─────────────────────────────────────────────────────────────
# 4.  Privacy accounting summary
# ─────────────────────────────────────────────────────────────
def privacy_report(engines: List, round_id: int):
    """Print per-client privacy budget consumed so far."""
    print(f"\n  Privacy budget after round {round_id}:")
    for i, eng in enumerate(engines):
        if eng is not None:
            eps, delta = get_privacy_spent(eng)
            print(f"    Client {i}: ε={eps:.3f}, δ={delta:.1e}")
