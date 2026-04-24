# algorithms/__init__.py
from .fedavg   import FedAvgClient
from .fedprox  import FedProxClient
from .scaffold import SCAFFOLDClient, SCAFFOLDServer
from .fednova  import FedNovaClient
from .dpfedavg import DPFedAvgClient

REGISTRY = {
    "FedAvg"   : FedAvgClient,
    "FedProx"  : FedProxClient,
    "SCAFFOLD" : SCAFFOLDClient,
    "FedNova"  : FedNovaClient,
    "DP-FedAvg": DPFedAvgClient,
}

# ══════════════════════════════════════════════════════════════
# algorithms/base.py  — shared client skeleton
# ══════════════════════════════════════════════════════════════
# algorithms/base.py
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
import numpy as np
from privacy import clip_gradients


class BaseClient:
    """
    Common FL client interface.
    Each algorithm subclass overrides `local_train()`.
    """

    def __init__(self, cid, model, train_loader, val_loader,
                 criterion, device, cfg_train, cfg_privacy):
        self.cid          = cid
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.device       = device
        self.cfg          = cfg_train
        self.priv         = cfg_privacy

    def get_parameters(self) -> List[np.ndarray]:
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params: List[np.ndarray]):
        sd = self.model.state_dict()
        for k, v in zip(sd.keys(), params):
            sd[k] = torch.tensor(v, device=self.device)
        self.model.load_state_dict(sd)

    def local_train(self, global_params, **kwargs) -> List[np.ndarray]:
        raise NotImplementedError

    def _make_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
        )


# ══════════════════════════════════════════════════════════════
# algorithms/fedavg.py
# ══════════════════════════════════════════════════════════════
# from .base import BaseClient
import copy, torch
import numpy as np
from typing import List

class FedAvgClient(BaseClient):
    """
    Standard FedAvg local training.
    McMahan et al. "Communication-Efficient Learning of Deep
    Networks from Decentralized Data" (AISTATS 2017).
    """

    def local_train(self, global_params: List[np.ndarray],
                    local_epochs: int = 3, **kwargs) -> List[np.ndarray]:
        self.set_parameters(global_params)
        opt = self._make_optimizer()
        self.model.train()

        for epoch in range(local_epochs):
            pbar = tqdm(self.train_loader, leave=False,
                        desc=f"    Client {self.cid} epoch {epoch+1}/{local_epochs}")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                opt.zero_grad()
                loss = self.criterion(self.model(imgs), labels)
                loss.backward()
                clip_gradients(self.model, self.priv.clip_norm)
                opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return self.get_parameters()


# ══════════════════════════════════════════════════════════════
# algorithms/fedprox.py
# ══════════════════════════════════════════════════════════════
class FedProxClient(BaseClient):
    """
    FedProx adds a proximal term μ/2 ||w - w_global||² to the local
    objective, penalising large deviations from the global model.
    Li et al. "Federated Optimization in Heterogeneous Networks"
    (MLSys 2020).
    """

    def local_train(self, global_params: List[np.ndarray],
                    local_epochs: int = 3, **kwargs) -> List[np.ndarray]:
        self.set_parameters(global_params)
        opt = self._make_optimizer()

        global_tensors = [torch.tensor(p, device=self.device)
                          for p in global_params]
        self.model.train()

        for epoch in range(local_epochs):
            pbar = tqdm(self.train_loader, leave=False,
                        desc=f"    Client {self.cid} epoch {epoch+1}/{local_epochs}")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                opt.zero_grad()
                ce_loss = self.criterion(self.model(imgs), labels)
                prox = sum(
                    ((p - g) ** 2).sum()
                    for p, g in zip(self.model.parameters(), global_tensors)
                )
                loss = ce_loss + (self.cfg.mu / 2) * prox
                loss.backward()
                clip_gradients(self.model, self.priv.clip_norm)
                opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 ce=f"{ce_loss.item():.4f}")

        return self.get_parameters()


# ══════════════════════════════════════════════════════════════
# algorithms/scaffold.py
# ══════════════════════════════════════════════════════════════
class SCAFFOLDClient(BaseClient):
    """
    SCAFFOLD corrects client drift via control variates.
    Each client maintains a local control variate c_i;
    the server maintains a global variate c.

    Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging
    for Federated Learning" (ICML 2020).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialise control variates to zero
        self.c_local  = [np.zeros_like(p) for p in self.get_parameters()]
        self.c_global = [np.zeros_like(p) for p in self.get_parameters()]

    def local_train(self, global_params : List[np.ndarray],
                    c_global            : List[np.ndarray],
                    local_epochs        : int = 3,
                    **kwargs) -> dict:
        self.set_parameters(global_params)
        self.c_global = c_global
        opt = self._make_optimizer()
        self.model.train()
        steps = 0

        for epoch in range(local_epochs):
            pbar = tqdm(self.train_loader, leave=False,
                        desc=f"    Client {self.cid} epoch {epoch+1}/{local_epochs}")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                opt.zero_grad()
                self.criterion(self.model(imgs), labels).backward()

                for param, ci, c in zip(self.model.parameters(),
                                        self.c_local, self.c_global):
                    if param.grad is not None:
                        param.grad.data -= (
                            torch.tensor(ci, device=self.device) -
                            torch.tensor(c,  device=self.device)
                        )
                clip_gradients(self.model, self.priv.clip_norm)
                opt.step()
                steps += 1
                pbar.set_postfix(step=steps)

        # Update local control variate (Option II)
        new_params   = self.get_parameters()
        lr, K        = self.cfg.lr, steps
        c_local_new  = [
            ci - c + (gp - lp) / (K * lr)
            for ci, c, gp, lp in zip(
                self.c_local, self.c_global, global_params, new_params)
        ]
        delta_c = [cn - co for cn, co in zip(c_local_new, self.c_local)]
        self.c_local = c_local_new

        return {"params": new_params, "delta_c": delta_c,
                "n_samples": len(self.train_loader.dataset)}


class SCAFFOLDServer:
    """Maintains the global control variate and aggregates delta_c."""

    def __init__(self, num_params: int):
        self.c_global = None   # initialised on first round

    def update_control_variate(self,
                                delta_cs    : List[List[np.ndarray]],
                                num_clients : int):
        if self.c_global is None:
            self.c_global = [np.zeros_like(d) for d in delta_cs[0]]
        for k in range(len(self.c_global)):
            self.c_global[k] += sum(d[k] for d in delta_cs) / num_clients


# ══════════════════════════════════════════════════════════════
# algorithms/fednova.py
# ══════════════════════════════════════════════════════════════
class FedNovaClient(BaseClient):
    """
    FedNova normalises local updates by the effective number of
    local steps, eliminating objective inconsistency caused by
    heterogeneous local training.

    Wang et al. "Tackling the Objective Inconsistency Problem in
    Heterogeneous Federated Optimization" (NeurIPS 2020).
    """

    def local_train(self, global_params : List[np.ndarray],
                    local_epochs        : int = 3,
                    **kwargs) -> dict:
        self.set_parameters(global_params)
        opt   = self._make_optimizer()
        rho   = self.cfg.rho         # local momentum factor
        tau   = 0                    # step counter
        self.model.train()

        # Momentum buffer for normalisation
        a_i = [np.zeros_like(p) for p in global_params]

        for epoch in range(local_epochs):
            pbar = tqdm(self.train_loader, leave=False,
                        desc=f"    Client {self.cid} epoch {epoch+1}/{local_epochs}")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                opt.zero_grad()
                self.criterion(self.model(imgs), labels).backward()
                clip_gradients(self.model, self.priv.clip_norm)
                opt.step()

                for k in range(len(a_i)):
                    a_i[k] = rho * a_i[k] + 1.0
                tau += 1
                pbar.set_postfix(tau=tau)

        local_params = self.get_parameters()
        # Normalised update: Δ_i / a_i
        norm_update = [
            (lp - gp) / (ai + 1e-10)
            for lp, gp, ai in zip(local_params, global_params, a_i)
        ]
        coeff = (1 - rho ** tau) / (1 - rho + 1e-10) if rho != 0 else tau

        return {"norm_update": norm_update,
                "coeff"      : coeff,
                "n_samples"  : len(self.train_loader.dataset)}


def fednova_aggregate(global_params   : List[np.ndarray],
                      client_results  : List[dict]) -> List[np.ndarray]:
    """Server-side FedNova aggregation."""
    total_coeff = sum(r["coeff"] for r in client_results)
    new_params  = [gp.copy() for gp in global_params]
    for r in client_results:
        w = r["coeff"] / total_coeff
        for k in range(len(new_params)):
            new_params[k] += w * r["norm_update"][k]
    return new_params


# ══════════════════════════════════════════════════════════════
# algorithms/dpfedavg.py
# ══════════════════════════════════════════════════════════════
class DPFedAvgClient(BaseClient):
    """
    DP-FedAvg: FedAvg with Opacus DP-SGD on each client.
    Each client runs DP-SGD locally; the server aggregates
    with plain weighted averaging (noise already added locally).

    Geyer et al. "Differentially Private Federated Learning:
    A Client Level Perspective" (2017).
    McMahan et al. "Learning Differentially Private Recurrent
    Language Models" (ICLR 2018).
    """

    def local_train(self, global_params : List[np.ndarray],
                    local_epochs        : int = 3,
                    **kwargs) -> dict:
        from privacy import make_private, get_privacy_spent

        self.set_parameters(global_params)
        opt = self._make_optimizer()

        # Wrap with Opacus
        priv_model, priv_opt, priv_loader, engine = make_private(
            self.model, opt, self.train_loader, self.priv
        )
        priv_model.train()

        for _ in range(local_epochs):
            for imgs, labels in priv_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                priv_opt.zero_grad()
                self.criterion(priv_model(imgs), labels).backward()
                priv_opt.step()

        eps, delta = get_privacy_spent(engine)
        # Sync weights back (Opacus wraps the original model in-place)
        self.model = priv_model._module

        return {"params"  : self.get_parameters(),
                "epsilon" : eps,
                "delta"   : delta,
                "n_samples": len(self.train_loader.dataset)}
