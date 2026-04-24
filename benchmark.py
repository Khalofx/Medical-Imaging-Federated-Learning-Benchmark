# benchmark.py
"""
Exhaustive benchmark runner with HPC-safe checkpointing.

Sweeps every combination of:
  - FL algorithm  × {FedAvg, FedProx, SCAFFOLD, FedNova, DP-FedAvg}
  - Dirichlet α   × {0.1, 0.5, 1.0, 5.0}

Checkpointing strategy
──────────────────────
Two granularities of state are saved:

  1. Per-round checkpoint  (every N rounds, configurable)
     checkpoints/{algo}/alpha{α}_round{r}.pt
       • global model weights
       • SCAFFOLD control variates  (if applicable)
       • round history so far
       • RNG states  (torch + numpy + python random)

  2. Experiment registry   results/completed_experiments.json
       • Tracks which (algo, α) pairs are fully done.
       • On resume, completed pairs are skipped entirely.
       • Partially-done pairs resume from the latest saved round.

On any restart benchmark.py automatically detects and resumes.

Usage:
    python benchmark.py                     # full sweep / auto-resume
    python benchmark.py --algo FedAvg       # single algorithm
    python benchmark.py --alpha 0.1         # single heterogeneity level
    python benchmark.py --reset             # wipe checkpoints, start fresh
"""
import argparse, copy, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from config     import DATA, FED, MODEL, TRAIN, PRIVACY, BENCH
from dataset    import (patient_level_split, dirichlet_partition,
                        scanner_profiles, make_loaders, get_pos_weight,
                        CXRBinaryDataset, base_transforms, build_image_index)
from models     import build_vit
from evaluate   import evaluate_model, metrics_report
from privacy    import SecureAggregator
from algorithms import REGISTRY, SCAFFOLDServer, fednova_aggregate
from visualize  import plot_all

import warnings; warnings.filterwarnings("ignore")

# Save a per-round checkpoint every N rounds (tune to your HPC wall-time)
CKPT_EVERY_N_ROUNDS = 1   # save after every round (set higher to reduce I/O)
BASELINE_ALGOS = ("Central", "Local-Only")

UNSUPPORTED_ALGOS = {
    "DP-FedAvg": (
        "Skipped: Opacus + ViT/timm attention stack is incompatible in the "
        "current software environment."
    ),
}


# ─────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────
def _ckpt_dir(algo: str) -> Path:
    p = Path(BENCH.checkpoint_dir) / algo
    p.mkdir(parents=True, exist_ok=True)
    return p

def _ckpt_path(algo: str, alpha: float, rnd: int) -> Path:
    return _ckpt_dir(algo) / f"alpha{alpha}_round{rnd:04d}.pt"

def _splits_path() -> Path:
    Path(BENCH.results_dir).mkdir(exist_ok=True)
    return Path(BENCH.results_dir) / "data_splits.pt"


def _registry_path() -> Path:
    Path(BENCH.results_dir).mkdir(exist_ok=True)
    return Path(BENCH.results_dir) / "completed_experiments.json"


def _skipped_path() -> Path:
    Path(BENCH.results_dir).mkdir(exist_ok=True)
    return Path(BENCH.results_dir) / "skipped_experiments.json"


def save_splits(train_df      : pd.DataFrame,
                val_df        : pd.DataFrame,
                test_df       : pd.DataFrame,
                partitions    : list,
                scanners      : list,
                alpha         : float,
                algo          : str):
    """
    Persist the exact DataFrames (indices + labels) used for this
    (algo, α) pair so a resumed run loads identical data.
    Scanners are saved too — feature-shift profiles must be
    consistent across sessions.
    """
    path = _splits_path()
    # Load existing splits file (may have entries for other combos)
    store = (
        torch.load(path, map_location="cpu", weights_only=False)
        if path.exists() else {}
    )
    key   = _exp_key(algo, alpha)
    store[key] = dict(
        train_df   = train_df.reset_index(drop=True),
        val_df     = val_df.reset_index(drop=True),
        test_df    = test_df.reset_index(drop=True),
        partitions = [p.reset_index(drop=True) for p in partitions],
        scanners   = scanners,
    )
    torch.save(store, path)
    print(f"  Splits saved → {path}")


def load_splits(algo: str, alpha: float) -> dict | None:
    """
    Return the saved split dict for this (algo, α) pair, or None
    if this is a fresh run.
    """
    path = _splits_path()
    if not path.exists():
        return None
    store = torch.load(path, map_location="cpu", weights_only=False)
    key   = _exp_key(algo, alpha)
    if key not in store:
        return None
    print(f"  Loaded saved splits for {algo} α={alpha}")
    return store[key]


def _load_registry() -> dict:
    p = _registry_path()
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}   # {"{algo}_{alpha}": {"test_metrics": ..., "history": [...]}}


def _save_registry(registry: dict):
    with open(_registry_path(), "w") as f:
        json.dump(registry, f, indent=2)


def _load_skipped_registry() -> dict:
    p = _skipped_path()
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _save_skipped_registry(registry: dict):
    with open(_skipped_path(), "w") as f:
        json.dump(registry, f, indent=2)


def _exp_key(algo: str, alpha: float) -> str:
    return f"{algo}_alpha{alpha}"


def _baseline_key(algo: str, alpha: float | None = None) -> str:
    return algo if alpha is None else f"{algo}_alpha{alpha}"


def load_any_saved_splits(alpha: float) -> dict | None:
    path = _splits_path()
    if not path.exists():
        return None
    store = torch.load(path, map_location="cpu", weights_only=False)
    suffix = f"_alpha{alpha}"
    for key in sorted(store):
        if key.endswith(suffix):
            print(f"  Loaded saved splits for α={alpha} from {key}")
            return store[key]
    return None


def save_round_checkpoint(algo_name   : str,
                           alpha       : float,
                           rnd         : int,
                           global_model: nn.Module,
                           history     : list,
                           best_round  : int | None = None,
                           best_metrics : dict | None = None,
                           best_state   : dict | None = None,
                           scaffold_server = None):
    """
    Persist everything needed to resume from this round.
    Called after every round (CKPT_EVERY_N_ROUNDS = 1).
    Keeps the two most recent round checkpoints to guard against
    a corrupt write on the very last save before a timeout.
    """
    payload = {
        "round"          : rnd,
        "model_state"    : global_model.state_dict(),
        "history"        : history,
        "best_round"     : best_round,
        "best_metrics"   : best_metrics,
        "best_state"     : best_state,
        "rng_torch"      : torch.get_rng_state(),
        "rng_torch_cuda" : torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None,
        "rng_numpy"      : np.random.get_state(),
        "rng_python"     : random.getstate(),
    }
    if scaffold_server is not None and scaffold_server.c_global is not None:
        payload["scaffold_c_global"] = scaffold_server.c_global

    path = _ckpt_path(algo_name, alpha, rnd)
    torch.save(payload, path)
    print(f"  ✓ Round checkpoint → {path.name}")

    # Keep only the two most recent round files to save disk space
    all_ckpts = sorted(_ckpt_dir(algo_name).glob(f"alpha{alpha}_round*.pt"))
    for old in all_ckpts[:-2]:
        old.unlink()


def load_latest_checkpoint(algo : str, alpha: float) -> dict | None:
    """
    Find the highest-round checkpoint for this (algo, α) pair.
    Returns the loaded payload dict, or None if no checkpoint exists.
    """
    ckpt_dir = _ckpt_dir(algo)
    pattern  = f"alpha{alpha}_round*.pt"
    files    = sorted(ckpt_dir.glob(pattern))
    if not files:
        return None
    latest = files[-1]
    print(f"  Resuming from checkpoint: {latest.name}")
    return torch.load(latest, map_location="cpu", weights_only=False)


def restore_rng_states(payload: dict):
    """Restore all RNG states for full reproducibility on resume."""
    torch.set_rng_state(payload["rng_torch"])
    if payload.get("rng_torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["rng_torch_cuda"])
    np.random.set_state(payload["rng_numpy"])
    random.setstate(payload["rng_python"])


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def fedavg_aggregate(global_params, client_params, n_samples):
    """Weighted FedAvg aggregation by dataset size."""
    total = sum(n_samples)
    agg   = [np.zeros_like(p) for p in global_params]
    for params, n in zip(client_params, n_samples):
        for k in range(len(agg)):
            agg[k] += (n / total) * params[k]
    return agg


def get_parameters(model): return [v.cpu().numpy() for v in model.state_dict().values()]
def set_parameters(model, params, device):
    sd = model.state_dict()
    for k, v in zip(sd.keys(), params):
        sd[k] = torch.tensor(v, device=device)
    model.load_state_dict(sd)


def _baseline_train_loader(df, image_index, scanner=None):
    scanner = scanner or {"gamma": 1.0, "blur": False}
    ds = CXRBinaryDataset(
        df,
        DATA.img_dir,
        transform=base_transforms(DATA.img_size, is_train=True, **scanner),
        image_index=image_index,
    )
    return DataLoader(
        ds,
        batch_size=FED.batch_size,
        shuffle=True,
        num_workers=DATA.num_workers,
        pin_memory=True,
    )


def _baseline_eval_loader(df, image_index):
    ds = CXRBinaryDataset(
        df,
        DATA.img_dir,
        transform=base_transforms(DATA.img_size, is_train=False),
        image_index=image_index,
    )
    return DataLoader(
        ds,
        batch_size=FED.batch_size,
        shuffle=False,
        num_workers=DATA.num_workers,
        pin_memory=True,
    )


def _train_one_epoch(model, loader, criterion, optimizer, device, desc):
    model.train()
    pbar = tqdm(loader, leave=False, desc=desc)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")


def _weighted_average_metrics(metrics: list[dict], weights: list[int]) -> dict:
    total = max(sum(weights), 1)
    out = {"n": int(total)}
    keys = [k for k in metrics[0].keys() if k != "n"]
    for key in keys:
        out[key] = sum(m[key] * w for m, w in zip(metrics, weights)) / total
    return out


def _baseline_epochs() -> int:
    # Keep baseline runtime modest while still giving a meaningful reference.
    return FED.num_rounds


def run_central_baseline(train_df, val_df, test_df, image_index: dict, device: str) -> dict:
    print(f"\n{'═'*60}")
    print("  Baseline: Central")
    print(f"{'═'*60}")

    train_loader = _baseline_train_loader(train_df, image_index)
    val_loader = _baseline_eval_loader(val_df, image_index)
    test_loader = _baseline_eval_loader(test_df, image_index)
    criterion = nn.BCEWithLogitsLoss(pos_weight=get_pos_weight(train_df).to(device))

    model = build_vit(MODEL.num_classes, MODEL.pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN.lr,
        weight_decay=TRAIN.weight_decay,
    )

    history = []
    best_round = None
    best_val_metrics = None
    best_model_state = None
    num_epochs = _baseline_epochs()

    epoch_pbar = tqdm(
        range(1, num_epochs + 1),
        total=num_epochs,
        desc="  [Central]",
        unit="epoch",
        leave=True,
    )
    for epoch in epoch_pbar:
        t0 = time.time()
        _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            desc=f"    Central epoch {epoch}/{num_epochs}",
        )

        m = evaluate_model(model, val_loader, criterion, device)
        history.append({"round": epoch, **m, "epsilon": np.nan})
        if best_val_metrics is None or m["auc"] > best_val_metrics["auc"]:
            best_round = epoch
            best_val_metrics = history[-1].copy()
            best_model_state = copy.deepcopy(model.state_dict())

        elapsed = time.time() - t0
        epoch_pbar.set_postfix(
            AUC=f"{m['auc']:.4f}",
            Sens=f"{m['sensitivity']:.4f}",
            Spec=f"{m['specificity']:.4f}",
            loss=f"{m['loss']:.4f}",
        )
        print(
            f"  Epoch {epoch:>2}/{num_epochs}  val_loss={m['loss']:.4f}  "
            f"AUC={m['auc']:.4f}  Sens={m['sensitivity']:.4f}  "
            f"Spec={m['specificity']:.4f}  ({elapsed:.1f}s)"
        )

    final_model_state = copy.deepcopy(model.state_dict())
    if best_model_state is None:
        best_model_state = copy.deepcopy(final_model_state)
        best_round = num_epochs
        best_val_metrics = history[-1].copy()
    else:
        model.load_state_dict(best_model_state)

    test_m = evaluate_model(model, test_loader, criterion, device)
    metrics_report(test_m, label=f"Central [TEST @ best val epoch {best_round}]")

    ckpt_dir = _ckpt_dir("Central")
    best_path = ckpt_dir / "best.pt"
    final_path = ckpt_dir / "final.pt"
    torch.save(best_model_state, best_path)
    torch.save(final_model_state, final_path)
    print(f"  Best model  → {best_path}")
    print(f"  Final model → {final_path}")

    return {
        "algo": "Central",
        "alpha": None,
        "history": history,
        "best_round": best_round,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_m,
    }


def run_local_only_baseline(
    alpha: float,
    train_df,
    val_df,
    test_df,
    scanners,
    image_index: dict,
    device: str,
) -> dict:
    print(f"\n{'═'*60}")
    print(f"  Baseline: Local-Only   α={alpha}")
    print(f"{'═'*60}")

    saved_splits = load_any_saved_splits(alpha)
    if saved_splits is not None:
        partitions = saved_splits["partitions"]
        scanners = saved_splits["scanners"]
        train_df = saved_splits["train_df"]
        val_df = saved_splits["val_df"]
        test_df = saved_splits["test_df"]
    else:
        partitions = dirichlet_partition(train_df, FED.num_clients, alpha, DATA.seed)
        save_splits(train_df, val_df, test_df, partitions, scanners, alpha, "Local-Only")

    val_loader = _baseline_eval_loader(val_df, image_index)
    test_loader = _baseline_eval_loader(test_df, image_index)
    eval_criterion = nn.BCEWithLogitsLoss(pos_weight=get_pos_weight(train_df).to(device))

    clients = []
    client_weights = [len(part) for part in partitions]
    for i, part in enumerate(partitions):
        model = build_vit(MODEL.num_classes, MODEL.pretrained).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAIN.lr,
            weight_decay=TRAIN.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=get_pos_weight(part).to(device))
        train_loader = _baseline_train_loader(part, image_index, scanners[i])
        clients.append({
            "cid": i,
            "model": model,
            "optimizer": optimizer,
            "criterion": criterion,
            "train_loader": train_loader,
        })

    history = []
    best_round = None
    best_val_metrics = None
    best_states = None
    num_epochs = _baseline_epochs()

    epoch_pbar = tqdm(
        range(1, num_epochs + 1),
        total=num_epochs,
        desc=f"  [Local-Only α={alpha}]",
        unit="epoch",
        leave=True,
    )
    for epoch in epoch_pbar:
        t0 = time.time()
        for client in clients:
            _train_one_epoch(
                client["model"],
                client["train_loader"],
                client["criterion"],
                client["optimizer"],
                device,
                desc=f"    Local client {client['cid']} epoch {epoch}/{num_epochs}",
            )

        val_metrics = [
            evaluate_model(client["model"], val_loader, eval_criterion, device)
            for client in clients
        ]
        agg = _weighted_average_metrics(val_metrics, client_weights)
        history.append({"round": epoch, **agg, "epsilon": np.nan})
        if best_val_metrics is None or agg["auc"] > best_val_metrics["auc"]:
            best_round = epoch
            best_val_metrics = history[-1].copy()
            best_states = [
                copy.deepcopy(client["model"].state_dict())
                for client in clients
            ]

        elapsed = time.time() - t0
        epoch_pbar.set_postfix(
            AUC=f"{agg['auc']:.4f}",
            Sens=f"{agg['sensitivity']:.4f}",
            Spec=f"{agg['specificity']:.4f}",
            loss=f"{agg['loss']:.4f}",
        )
        print(
            f"  Epoch {epoch:>2}/{num_epochs}  val_loss={agg['loss']:.4f}  "
            f"AUC={agg['auc']:.4f}  Sens={agg['sensitivity']:.4f}  "
            f"Spec={agg['specificity']:.4f}  ({elapsed:.1f}s)"
        )

    final_states = [copy.deepcopy(client["model"].state_dict()) for client in clients]
    if best_states is None:
        best_states = copy.deepcopy(final_states)
        best_round = num_epochs
        best_val_metrics = history[-1].copy()

    for client, state in zip(clients, best_states):
        client["model"].load_state_dict(state)

    test_metrics = [
        evaluate_model(client["model"], test_loader, eval_criterion, device)
        for client in clients
    ]
    test_m = _weighted_average_metrics(test_metrics, client_weights)
    metrics_report(
        test_m,
        label=f"Local-Only α={alpha} [TEST @ best val epoch {best_round}]",
    )

    ckpt_dir = _ckpt_dir("Local-Only")
    best_path = ckpt_dir / f"alpha{alpha}_best.pt"
    final_path = ckpt_dir / f"alpha{alpha}_final.pt"
    torch.save(
        {"client_states": best_states, "best_round": best_round},
        best_path,
    )
    torch.save(
        {"client_states": final_states, "best_round": best_round},
        final_path,
    )
    print(f"  Best model  → {best_path}")
    print(f"  Final model → {final_path}")

    return {
        "algo": "Local-Only",
        "alpha": alpha,
        "history": history,
        "best_round": best_round,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_m,
    }


# ─────────────────────────────────────────────────────────────
# Single experiment
# ─────────────────────────────────────────────────────────────
def run_experiment(algo_name : str,
                   alpha     : float,
                   train_df, val_df, test_df,
                   scanners,
                   image_index : dict,
                   device    : str,
) -> dict:
    print(f"\n{'═'*60}")
    print(f"  Algorithm: {algo_name}   α={alpha}")
    print(f"{'═'*60}")

    # ── Partition — load saved or create fresh ─────────────────
    saved_splits = load_splits(algo_name, alpha)
    if saved_splits is not None:
        # Resume path: use the exact same DataFrames as the first run
        partitions = saved_splits["partitions"]
        scanners   = saved_splits["scanners"]
        train_df   = saved_splits["train_df"]
        val_df     = saved_splits["val_df"]
        test_df    = saved_splits["test_df"]
    else:
        # Fresh run: partition now and immediately persist
        partitions = dirichlet_partition(
            train_df, FED.num_clients, alpha, DATA.seed)
        save_splits(train_df, val_df, test_df,
                    partitions, scanners, alpha, algo_name)
    pos_weight = get_pos_weight(train_df).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Build client loaders ──────────────────────────────────
    train_batch_size = (
        PRIVACY.dp_batch_size if algo_name == "DP-FedAvg"
        else FED.batch_size
    )
    if algo_name == "DP-FedAvg":
        print(f"  Using DP local batch size={train_batch_size}")
    loaders = [
        make_loaders(partitions[i], val_df, DATA.img_dir,
                     DATA.img_size, train_batch_size, DATA.num_workers,
                     scanners[i], image_index=image_index)
        for i in range(FED.num_clients)
    ]

    # ── Global model ───────────────────────────────────────────
    global_model = build_vit(MODEL.num_classes, MODEL.pretrained).to(device)
    if algo_name == "DP-FedAvg":
        disabled = 0
        for block in getattr(global_model, "blocks", []):
            attn = getattr(block, "attn", None)
            if hasattr(attn, "fused_attn"):
                attn.fused_attn = False
                disabled += 1
        if disabled:
            print(f"  Disabled fused attention in {disabled} ViT blocks for DP")

    # ── Algorithm-specific server state ───────────────────────
    scaffold_server = SCAFFOLDServer(None) if algo_name == "SCAFFOLD" else None
    secagg          = SecureAggregator(FED.num_clients, PRIVACY.secagg_bits)

    # ── Build clients ──────────────────────────────────────────
    ClientClass = REGISTRY[algo_name]
    clients = [
        ClientClass(
            cid          = i,
            model        = copy.deepcopy(global_model),
            train_loader = loaders[i][0],
            val_loader   = loaders[i][1],
            criterion    = criterion,
            device       = device,
            cfg_train    = TRAIN,
            cfg_privacy  = PRIVACY,
        )
        for i in range(FED.num_clients)
    ]

    # ── Resume from checkpoint if available ───────────────────
    ckpt    = load_latest_checkpoint(algo_name, alpha)
    history = []
    start_round = 1
    best_round = None
    best_val_metrics = None
    best_model_state = None

    if ckpt is not None:
        global_model.load_state_dict(ckpt["model_state"])
        history     = ckpt["history"]
        start_round = ckpt["round"] + 1
        best_round = ckpt.get("best_round")
        best_val_metrics = ckpt.get("best_metrics")
        best_model_state = ckpt.get("best_state")
        restore_rng_states(ckpt)
        if scaffold_server is not None and "scaffold_c_global" in ckpt:
            scaffold_server.c_global = ckpt["scaffold_c_global"]
        if ckpt["round"] >= FED.num_rounds:
            print(
                "  Checkpoint already completed all configured rounds; "
                "skipping training and running final evaluation."
            )
        else:
            print(f"  Resumed at round {start_round}/{FED.num_rounds}")
    else:
        if scaffold_server is not None:
            scaffold_server.c_global = [
                np.zeros_like(p) for p in get_parameters(global_model)]

    global_params = get_parameters(global_model)

    # ── Federated rounds ───────────────────────────────────────
    if start_round <= FED.num_rounds:
        round_pbar = tqdm(range(start_round, FED.num_rounds + 1),
                          initial=start_round - 1,
                          total=FED.num_rounds,
                          desc=f"  [{algo_name} α={alpha}]",
                          unit="round", position=0, leave=True)

        for rnd in round_pbar:
            t0 = time.time()
            client_results, n_samples = [], []

            for i, client in enumerate(clients):

                if algo_name == "SCAFFOLD":
                    result = client.local_train(
                        global_params, c_global=scaffold_server.c_global,
                        local_epochs=FED.local_epochs)
                    client_results.append(result)
                    n_samples.append(result["n_samples"])

                elif algo_name == "FedNova":
                    result = client.local_train(
                        global_params, local_epochs=FED.local_epochs)
                    client_results.append(result)
                    n_samples.append(result["n_samples"])

                elif algo_name == "DP-FedAvg":
                    result = client.local_train(
                        global_params, local_epochs=FED.local_epochs)
                    client_results.append(result)
                    n_samples.append(result["n_samples"])

                else:  # FedAvg, FedProx
                    params = client.local_train(
                        global_params, local_epochs=FED.local_epochs)
                    client_results.append({"params": params,
                                           "n_samples": len(loaders[i][0].dataset)})
                    n_samples.append(client_results[-1]["n_samples"])

            # ── Secure Aggregation masking (if enabled) ──────
            if algo_name == "FedNova":
                agg_params = fednova_aggregate(global_params, client_results)
            elif PRIVACY.secagg_enabled and algo_name != "DP-FedAvg":
                masked = [
                    secagg.mask_updates(i, r["params"], rnd)
                    for i, r in enumerate(client_results)
                ]
                agg_params = secagg.aggregate(masked)
            else:
                if algo_name == "SCAFFOLD":
                    raw_params = [r["params"] for r in client_results]
                    agg_params = fedavg_aggregate(global_params, raw_params, n_samples)
                    scaffold_server.update_control_variate(
                        [r["delta_c"] for r in client_results], FED.num_clients)
                else:
                    raw_params = [r["params"] for r in client_results]
                    agg_params = fedavg_aggregate(global_params, raw_params, n_samples)

            global_params = agg_params
            set_parameters(global_model, global_params, device)

            # ── Round evaluation on validation set ───────────
            val_ds  = CXRBinaryDataset(
                val_df, DATA.img_dir,
                transform=base_transforms(DATA.img_size, is_train=False),
                image_index=image_index)
            val_loader = DataLoader(val_ds, batch_size=FED.batch_size,
                                    shuffle=False, num_workers=DATA.num_workers)
            m = evaluate_model(global_model, val_loader, criterion, device)

            eps_str = ""
            if algo_name == "DP-FedAvg":
                epsilons = [r.get("epsilon", None) for r in client_results]
                eps_str  = f"  ε={np.mean([e for e in epsilons if e]):.2f}"

            elapsed = time.time() - t0
            round_pbar.set_postfix(
                AUC=f"{m['auc']:.4f}",
                Sens=f"{m['sensitivity']:.4f}",
                Spec=f"{m['specificity']:.4f}",
                loss=f"{m['loss']:.4f}",
            )
            print(f"  Round {rnd:>2}/{FED.num_rounds}  "
                  f"val_loss={m['loss']:.4f}  AUC={m['auc']:.4f}  "
                  f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}"
                  f"{eps_str}  ({elapsed:.1f}s)")

            history.append({"round": rnd, **m,
                            "epsilon": np.mean([r.get("epsilon", np.nan)
                                       for r in client_results])})

            if best_val_metrics is None or m["auc"] > best_val_metrics["auc"]:
                best_round = rnd
                best_val_metrics = history[-1].copy()
                best_model_state = copy.deepcopy(global_model.state_dict())
                print(
                    f"  Best checkpoint updated → round {best_round} "
                    f"(val AUC={m['auc']:.4f})"
                )

            # ── Save after every round ────────────────────────
            save_round_checkpoint(
                algo_name, alpha, rnd,
                global_model, history,
                best_round=best_round,
                best_metrics=best_val_metrics,
                best_state=best_model_state,
                scaffold_server=scaffold_server,
            )

    final_model_state = copy.deepcopy(global_model.state_dict())
    if best_model_state is None:
        best_model_state = copy.deepcopy(final_model_state)
        if history:
            best_round = history[-1]["round"]
            best_val_metrics = history[-1].copy()
    else:
        global_model.load_state_dict(best_model_state)

    # ── Best-checkpoint test evaluation ───────────────────────
    test_ds = CXRBinaryDataset(
        test_df, DATA.img_dir,
        transform=base_transforms(DATA.img_size, is_train=False),
        image_index=image_index)
    test_loader = DataLoader(test_ds, batch_size=FED.batch_size,
                             shuffle=False, num_workers=DATA.num_workers)
    test_m = evaluate_model(global_model, test_loader, criterion, device)
    metrics_report(
        test_m,
        label=f"{algo_name} α={alpha} [TEST @ best val round {best_round}]",
    )

    # ── Save final and best model weights ──────────────────────
    best_path = _ckpt_dir(algo_name) / f"alpha{alpha}_best.pt"
    final_path = _ckpt_dir(algo_name) / f"alpha{alpha}_final.pt"
    torch.save(best_model_state, best_path)
    torch.save(final_model_state, final_path)
    print(f"  Best model  → {best_path}")
    print(f"  Final model → {final_path}")

    return {
        "algo": algo_name,
        "alpha": alpha,
        "history": history,
        "best_round": best_round,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_m,
    }


# ─────────────────────────────────────────────────────────────
# Full benchmark sweep
# ─────────────────────────────────────────────────────────────
def run_benchmark(algos: list = None, alphas: list = None):
    import torch, os
    device = BENCH.device if torch.cuda.is_available() else "cpu"
    smoke  = os.environ.get("FL_SMOKE_TEST", "0") == "1"

    print("============================================")
    print(f"  Mode   : {'⚡ SMOKE TEST' if smoke else '🚀 FULL BENCHMARK'}")
    print(f"  Rounds : {FED.num_rounds}   Local epochs: {FED.local_epochs}")
    print(f"  Device : {device}")
    print("============================================")

    requested_algos = algos or list(BENCH.algorithms) + list(BASELINE_ALGOS)
    fl_algos = [a for a in requested_algos if a in BENCH.algorithms]
    want_central = "Central" in requested_algos
    want_local_only = "Local-Only" in requested_algos
    alphas = alphas or DATA.alpha_values

    # Load & split data once
    print("\nLoading dataset...")
    train_df, val_df, test_df = patient_level_split(DATA.csv_path, DATA.seed)
    scanners    = scanner_profiles(FED.num_clients, DATA.seed)
    image_index = build_image_index(DATA.img_dir)
    print(f"  Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    registry    = _load_registry()
    skipped_registry = _load_skipped_registry()
    all_results = []

    for key, val in registry.items():
        all_results.append(val)

    if want_central:
        key = _baseline_key("Central")
        if key in registry:
            print(f"\n  [SKIP] Central already complete.")
        else:
            res = run_central_baseline(
                train_df, val_df, test_df, image_index, device
            )
            all_results.append(res)
            registry[key] = res
            _save_registry(registry)
            print("  [DONE] Central → registry updated.")

    for alpha in alphas:
        if want_local_only:
            local_key = _baseline_key("Local-Only", alpha)
            if local_key in registry:
                print(f"\n  [SKIP] Local-Only α={alpha} already complete.")
            else:
                res = run_local_only_baseline(
                    alpha, train_df, val_df, test_df, scanners, image_index, device
                )
                all_results.append(res)
                registry[local_key] = res
                _save_registry(registry)
                print(f"  [DONE] Local-Only α={alpha} → registry updated.")

        for algo in fl_algos:
            key = _exp_key(algo, alpha)
            if algo in UNSUPPORTED_ALGOS:
                reason = UNSUPPORTED_ALGOS[algo]
                print(f"\n  [SKIP] {algo} α={alpha} unsupported. {reason}")
                skipped_registry[key] = {
                    "algorithm": algo,
                    "alpha": alpha,
                    "status": "skipped",
                    "note": reason,
                }
                _save_skipped_registry(skipped_registry)
                continue
            if key in registry:
                print(f"\n  [SKIP] {algo} α={alpha} already complete.")
                continue
            res = run_experiment(
                algo, alpha, train_df, val_df, test_df,
                scanners, image_index, device)
            all_results.append(res)

            # Mark as complete and persist immediately
            registry[key] = res
            _save_registry(registry)
            print(f"  [DONE] {algo} α={alpha} → registry updated.")

    # ── Save results table ────────────────────────────────────
    rows = []
    for r in all_results:
        tm = r["test_metrics"]
        rows.append({
            "algorithm"     : r["algo"],
            "alpha"         : r["alpha"],
            "status"        : "completed",
            "best_val_round": r.get("best_round", ""),
            "best_val_auc"  : round(
                r.get("best_val_metrics", {}).get("auc", np.nan), 4
            ) if r.get("best_val_metrics") else np.nan,
            "test_auc"      : round(tm["auc"],         4),
            "test_acc"      : round(tm["accuracy"],    4),
            "sensitivity"   : round(tm["sensitivity"], 4),
            "specificity"   : round(tm["specificity"], 4),
            "final_epsilon" : r["history"][-1].get("epsilon", "N/A")
                              if r["history"] else "N/A",
            "note"          : "",
        })
    for r in skipped_registry.values():
        rows.append({
            "algorithm"     : r["algorithm"],
            "alpha"         : r["alpha"],
            "status"        : r["status"],
            "best_val_round": "",
            "best_val_auc"  : np.nan,
            "test_auc"      : np.nan,
            "test_acc"      : np.nan,
            "sensitivity"   : np.nan,
            "specificity"   : np.nan,
            "final_epsilon" : "N/A",
            "note"          : r["note"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["algorithm", "alpha"], kind="stable").reset_index(drop=True)
    out = Path(BENCH.results_dir)
    out.mkdir(exist_ok=True)
    df.to_csv(out / "benchmark_results.csv", index=False)
    print(f"\nResults saved → {out / 'benchmark_results.csv'}")
    print(df.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────
    plot_all(all_results, out)
    return df


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",  type=str, default=None,
                        choices=list(REGISTRY.keys()) + list(BASELINE_ALGOS) + [None])
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--reset", action="store_true",
                        help="Delete all checkpoints and the registry, "
                             "then start from scratch.")
    args = parser.parse_args()

    if args.reset:
        import shutil
        print("⚠  --reset: deleting checkpoints, splits, and registries...")
        shutil.rmtree(BENCH.checkpoint_dir, ignore_errors=True)
        for f in [_registry_path(), _skipped_path(), _splits_path()]:
            if f.exists():
                f.unlink()
        print("   Done. Starting fresh.\n")

    run_benchmark(
        algos  = [args.algo]  if args.algo  else None,
        alphas = [args.alpha] if args.alpha else None,
    )
