from typing import List

import numpy as np
import torch
from privacy import clip_gradients
from tqdm import tqdm

from .fedavg import _BaseClient


class SCAFFOLDClient(_BaseClient):
    """
    SCAFFOLD corrects client drift via control variates.
    Each client maintains a local control variate c_i;
    the server maintains a global variate c.

    Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging
    for Federated Learning" (ICML 2020).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_local = [np.zeros_like(p) for p in self.get_parameters()]
        self.c_global = [np.zeros_like(p) for p in self.get_parameters()]

    def local_train(
        self,
        global_params: List[np.ndarray],
        c_global: List[np.ndarray],
        local_epochs: int = 3,
        **kwargs,
    ) -> dict:
        self.set_parameters(global_params)
        self.c_global = c_global
        opt = self._make_optimizer()
        self.model.train()
        steps = 0

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

                for param, ci, c in zip(
                    self.model.parameters(),
                    self.c_local,
                    self.c_global,
                ):
                    if param.grad is not None:
                        param.grad.data -= (
                            torch.tensor(ci, device=self.device)
                            - torch.tensor(c, device=self.device)
                        )
                clip_gradients(self.model, self.priv.clip_norm)
                opt.step()
                steps += 1
                pbar.set_postfix(step=steps)

        new_params = self.get_parameters()
        lr, k_steps = self.cfg.lr, steps
        c_local_new = [
            ci - c + (gp - lp) / (k_steps * lr)
            for ci, c, gp, lp in zip(
                self.c_local,
                self.c_global,
                global_params,
                new_params,
            )
        ]
        delta_c = [cn - co for cn, co in zip(c_local_new, self.c_local)]
        self.c_local = c_local_new

        return {
            "params": new_params,
            "delta_c": delta_c,
            "n_samples": len(self.train_loader.dataset),
        }


class SCAFFOLDServer:
    """Maintains the global control variate and aggregates delta_c."""

    def __init__(self, num_params: int):
        self.c_global = None

    def update_control_variate(
        self,
        delta_cs: List[List[np.ndarray]],
        num_clients: int,
    ):
        if self.c_global is None:
            self.c_global = [np.zeros_like(d) for d in delta_cs[0]]
        for k in range(len(self.c_global)):
            self.c_global[k] += sum(d[k] for d in delta_cs) / num_clients
