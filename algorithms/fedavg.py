from typing import List

import numpy as np
import torch
from privacy import clip_gradients
from tqdm import tqdm


class _BaseClient:
    """
    Common FL client interface.
    Each algorithm subclass overrides `local_train()`.
    """

    def __init__(
        self,
        cid,
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        cfg_train,
        cfg_privacy,
    ):
        self.cid = cid
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.cfg = cfg_train
        self.priv = cfg_privacy

    def get_parameters(self) -> List[np.ndarray]:
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params: List[np.ndarray]):
        sd = self.model.state_dict()
        for k, v in zip(sd.keys(), params):
            sd[k] = torch.tensor(v, device=self.device)
        self.model.load_state_dict(sd)

    def _make_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )


class FedAvgClient(_BaseClient):
    """
    Standard FedAvg local training.
    McMahan et al. "Communication-Efficient Learning of Deep
    Networks from Decentralized Data" (AISTATS 2017).
    """

    def local_train(
        self,
        global_params: List[np.ndarray],
        local_epochs: int = 3,
        **kwargs,
    ) -> List[np.ndarray]:
        self.set_parameters(global_params)
        opt = self._make_optimizer()
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
                loss = self.criterion(self.model(imgs), labels)
                loss.backward()
                clip_gradients(self.model, self.priv.clip_norm)
                opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return self.get_parameters()
