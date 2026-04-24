from typing import List

import numpy as np
from privacy import clip_gradients
from tqdm import tqdm

from .fedavg import _BaseClient


class FedNovaClient(_BaseClient):
    """
    FedNova normalises local updates by the effective number of
    local steps, eliminating objective inconsistency caused by
    heterogeneous local training.

    Wang et al. "Tackling the Objective Inconsistency Problem in
    Heterogeneous Federated Optimization" (NeurIPS 2020).
    """

    def local_train(
        self,
        global_params: List[np.ndarray],
        local_epochs: int = 3,
        **kwargs,
    ) -> dict:
        self.set_parameters(global_params)
        opt = self._make_optimizer()
        rho = self.cfg.rho
        tau = 0
        self.model.train()

        a_i = [np.zeros_like(p) for p in global_params]

        for epoch in range(local_epochs):
            pbar = tqdm(
                self.train_loader,
                leave=False,
                desc=f"    Client {self.cid} epoch {epoch + 1}/{local_epochs}",
            )
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
        norm_update = [
            (lp - gp) / (ai + 1e-10)
            for lp, gp, ai in zip(local_params, global_params, a_i)
        ]
        coeff = (1 - rho**tau) / (1 - rho + 1e-10) if rho != 0 else tau

        return {
            "norm_update": norm_update,
            "coeff": coeff,
            "n_samples": len(self.train_loader.dataset),
        }


def fednova_aggregate(
    global_params: List[np.ndarray],
    client_results: List[dict],
) -> List[np.ndarray]:
    """Server-side FedNova aggregation."""
    total_coeff = sum(r["coeff"] for r in client_results)
    new_params = [gp.copy() for gp in global_params]
    for r in client_results:
        weight = r["coeff"] / total_coeff
        for k in range(len(new_params)):
            new_params[k] += weight * r["norm_update"][k]
    return new_params
