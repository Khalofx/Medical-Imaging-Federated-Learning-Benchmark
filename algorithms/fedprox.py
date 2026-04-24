from typing import List

import numpy as np
import torch
from privacy import clip_gradients
from tqdm import tqdm

from .fedavg import _BaseClient


class FedProxClient(_BaseClient):
    """
    FedProx adds a proximal term μ/2 ||w - w_global||² to the local
    objective, penalising large deviations from the global model.
    Li et al. "Federated Optimization in Heterogeneous Networks"
    (MLSys 2020).
    """

    def local_train(
        self,
        global_params: List[np.ndarray],
        local_epochs: int = 3,
        **kwargs,
    ) -> List[np.ndarray]:
        self.set_parameters(global_params)
        opt = self._make_optimizer()

        global_tensors = [torch.tensor(p, device=self.device) for p in global_params]
        self.model.train()

        for epoch in range(local_epochs):
            pbar = tqdm(
                self.train_loader,
                leave=False,
                desc=f"    Client {self.cid} epoch {epoch + 1}/{local_epochs}",
            )
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
                pbar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{ce_loss.item():.4f}")

        return self.get_parameters()
