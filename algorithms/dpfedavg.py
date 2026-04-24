from typing import List

import numpy as np

from .fedavg import _BaseClient


class DPFedAvgClient(_BaseClient):
    """
    DP-FedAvg: FedAvg with Opacus DP-SGD on each client.
    Each client runs DP-SGD locally; the server aggregates
    with plain weighted averaging (noise already added locally).

    Geyer et al. "Differentially Private Federated Learning:
    A Client Level Perspective" (2017).
    McMahan et al. "Learning Differentially Private Recurrent
    Language Models" (ICLR 2018).
    """

    def local_train(
        self,
        global_params: List[np.ndarray],
        local_epochs: int = 3,
        **kwargs,
    ) -> dict:
        from privacy import get_privacy_spent, make_private

        self.set_parameters(global_params)
        opt = self._make_optimizer()

        priv_model, priv_opt, priv_loader, engine = make_private(
            self.model,
            opt,
            self.train_loader,
            self.priv,
        )
        priv_model.train()

        for _ in range(local_epochs):
            for imgs, labels in priv_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                priv_opt.zero_grad()
                self.criterion(priv_model(imgs), labels).backward()
                priv_opt.step()

        eps, delta = get_privacy_spent(engine)
        self.model = priv_model._module

        return {
            "params": self.get_parameters(),
            "epsilon": eps,
            "delta": delta,
            "n_samples": len(self.train_loader.dataset),
        }
