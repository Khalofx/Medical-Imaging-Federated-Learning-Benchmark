# FL Chest X-ray Benchmark

Federated learning benchmark for binary chest X-ray classification on the NIH ChestX-ray14 dataset using a Vision Transformer (ViT) backbone. The project compares multiple federated optimization strategies under label-distribution heterogeneity and simulated scanner-domain shift, with support for checkpointed HPC runs, post-hoc analysis plots, and reproducible experiment resumption.

## Overview

This repository studies how federated learning methods behave on a medical imaging task where each client represents a different hospital or site. The benchmark:

- uses the NIH ChestX-ray14 dataset,
- converts the task to binary classification: `No Finding` vs `Suspicious`,
- splits data at the patient level to avoid leakage,
- creates non-IID client partitions with a Dirichlet distribution,
- optionally simulates scanner differences with brightness and blur shifts,
- trains a ViT-based classifier across multiple FL algorithms,
- evaluates accuracy, AUC, sensitivity, specificity, F1, and calibration,
- saves checkpoints and split metadata so interrupted runs can resume safely.

## Implemented Methods

Supported baselines and federated methods:

- `Central`: centralized training on the pooled training set
- `Local-Only`: independent client training without aggregation
- `FedAvg`
- `FedProx`
- `SCAFFOLD`
- `FedNova`
- `DP-FedAvg`

## Important Limitation

`DP-FedAvg` is implemented in the codebase, but the benchmark currently skips it at runtime because the present software stack reports an incompatibility between Opacus and the ViT/timm attention setup used here. The skip is recorded in `results/skipped_experiments.json`.

## Dataset

This project expects the NIH ChestX-ray14 dataset to be available locally.

Required files:

- `Data_Entry_2017.csv`
- the image folders from the NIH release, typically arranged as `images_001/images`, `images_002/images`, ..., `images_012/images`

Current default paths are configured in [`config.py`](/home/khalifa.juma/HC701/Project/config.py):

- `DATA.data_dir = /home/khalifa.juma/HC701/Project/NIH_Chest_Xrays`
- `DATA.img_dir = /home/khalifa.juma/HC701/Project/NIH_Chest_Xrays`
- `DATA.csv_path = /home/khalifa.juma/HC701/Project/NIH_Chest_Xrays/Data_Entry_2017.csv`

Expected layout:

```text
NIH_Chest_Xrays/
├── Data_Entry_2017.csv
├── images_001/
│   └── images/
├── images_002/
│   └── images/
├── ...
└── images_012/
    └── images/
```

## Problem Setup

### Task

Binary classification:

- `0`: `No Finding`
- `1`: any other finding, grouped as `Suspicious`

### Data Splitting

- patient-level split to prevent leakage,
- `70%` train / `15%` validation / `15%` test,
- reproducible through a fixed random seed.

### Federated Partitioning

Training data is divided across `5` clients by default using a Dirichlet distribution over binary labels.

- lower `alpha` means more non-IID client label distributions,
- higher `alpha` means more IID-like distributions.

Default sweep values:

- `0.1`
- `0.5`
- `1.0`
- `5.0`

### Scanner Shift Simulation

Each client can be assigned a synthetic scanner profile with:

- a brightness multiplier (`gamma`),
- optional Gaussian blur.

This simulates site-specific acquisition differences across hospitals.

## Model

The benchmark uses a Vision Transformer classifier:

- architecture: `vit_base_patch16_224`
- pretrained ImageNet initialization by default
- binary output head (`num_classes = 1`)

Model construction is handled in [`models.py`](/home/khalifa.juma/HC701/Project/models.py).

## Evaluation Metrics

The benchmark reports:

- loss
- AUC
- accuracy
- sensitivity
- specificity
- F1 score
- Brier score

Evaluation utilities live in [`evaluate.py`](/home/khalifa.juma/HC701/Project/evaluate.py).

## Repository Structure

```text
Project/
├── README.md
├── .gitignore
├── environment.yml
├── config.py
├── benchmark.py
├── dataset.py
├── models.py
├── evaluate.py
├── privacy.py
├── visualize.py
├── generate_fairness_plots.py
├── generate_roc_plot.py
├── run_benchmark.sh
├── run_smoke_test.sh
├── run_analysis_plots.sh
├── algorithms/
│   ├── fedavg.py
│   ├── fedprox.py
│   ├── scaffold.py
│   ├── fednova.py
│   └── dpfedavg.py
├── checkpoints/
├── results/
├── logs_slurm/
└── NIH_Chest_Xrays/
```

## Environment Setup

Create the conda environment from [`environment.yml`](/home/khalifa.juma/HC701/Project/environment.yml):

```bash
cd /home/khalifa.juma/HC701/Project
conda env create -f environment.yml
conda activate fl_cxr
```

Main dependencies:

- Python 3.10
- PyTorch 2.2
- torchvision 0.17
- timm
- Flower
- Opacus
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm

If you are running on CPU only, you may need to adapt the CUDA-specific dependency in `environment.yml`.

## Configuration

Most experiment settings are centralized in [`config.py`](/home/khalifa.juma/HC701/Project/config.py).

Key defaults:

- `num_clients = 5`
- `num_rounds = 20`
- `local_epochs = 3`
- `batch_size = 32`
- `img_size = 224`
- `lr = 1e-4`
- `weight_decay = 1e-2`
- `results_dir = ./results`
- `checkpoint_dir = ./checkpoints`

Smoke-test mode is enabled by setting:

```bash
export FL_SMOKE_TEST=1
```

When smoke mode is on, the config switches to:

- `2` rounds
- `1` local epoch

## Running the Benchmark

### Full Benchmark

```bash
cd /home/khalifa.juma/HC701/Project
conda activate fl_cxr
python benchmark.py
```

This runs the full sweep across:

- algorithms: `FedAvg`, `FedProx`, `SCAFFOLD`, `FedNova`, `DP-FedAvg`
- heterogeneity levels: `0.1`, `0.5`, `1.0`, `5.0`

Centralized and local-only baselines are also included in the benchmark pipeline.

### Run a Single Algorithm

```bash
python benchmark.py --algo FedAvg
```

### Run a Single Heterogeneity Level

```bash
python benchmark.py --alpha 0.5
```

### Run One Algorithm at One Alpha

```bash
python benchmark.py --algo SCAFFOLD --alpha 1.0
```

### Reset and Start Fresh

```bash
python benchmark.py --reset
```

This removes:

- benchmark checkpoints,
- saved split metadata,
- completed experiment registry,
- skipped experiment registry.

## HPC / SLURM Usage

### Benchmark Job

Use [`run_benchmark.sh`](/home/khalifa.juma/HC701/Project/run_benchmark.sh):

```bash
sbatch run_benchmark.sh
```

This script:

- requests 1 GPU,
- activates the `fl_cxr` conda environment,
- exports `NUM_WORKERS` from the SLURM CPU allocation,
- checks required paths,
- runs `benchmark.py` with automatic resume behavior.

You can also forward CLI arguments:

```bash
sbatch run_benchmark.sh --algo FedAvg --alpha 1.0
```

### Smoke Test

Use [`run_smoke_test.sh`](/home/khalifa.juma/HC701/Project/run_smoke_test.sh):

```bash
sbatch run_smoke_test.sh
```

This launches a minimal validation run:

- algorithm: `FedAvg`
- alpha: `1.0`
- rounds: `2`
- local epochs: `1`

### Analysis Plots

After benchmark checkpoints and registries exist, run:

```bash
sbatch run_analysis_plots.sh
```

This script generates:

- fairness plots,
- ROC curves,
- additional result summaries under `results/`.

## Checkpointing and Resume Logic

The benchmark is designed for long HPC jobs and interrupted runs.

It saves:

- per-round checkpoints under `checkpoints/<algorithm>/`,
- split metadata in `results/data_splits.pt`,
- completed experiments in `results/completed_experiments.json`,
- skipped experiments in `results/skipped_experiments.json`.

Behavior:

- completed `(algorithm, alpha)` combinations are skipped on rerun,
- partially completed runs resume from the latest round checkpoint,
- RNG states are restored for reproducibility after resume.

## Outputs

### Main Result Files

The benchmark writes:

- `results/benchmark_results.csv`
- `results/completed_experiments.json`
- `results/skipped_experiments.json`
- `results/data_splits.pt`

### Generated Plots

The repository includes scripts to generate or regenerate:

- `results/convergence_curves.png`
- `results/auc_heatmap.png`
- `results/test_auc_vs_alpha.png`
- `results/best_round_heatmap.png`
- `results/brier_heatmap.png`
- `results/clinical_metrics_heatmaps.png`
- `results/data_split_overview.png`
- `results/roc_curves.png`
- `results/fairness_hospital_auc.png`
- `results/fairness_auc_gap_heatmap.png`
- `results/fairness_best_worst_auc.png`
- `results/fairness_hospital_metrics.csv`
- `results/fairness_best_worst_hospitals.csv`

## Privacy Components

Privacy-related utilities are implemented in [`privacy.py`](/home/khalifa.juma/HC701/Project/privacy.py):

- DP-SGD via Opacus,
- simulated secure aggregation,
- gradient clipping,
- privacy budget reporting.

At the moment, only the code path exists for `DP-FedAvg`; benchmark execution currently skips it due to the ViT/Opacus compatibility issue described above.

## Reproducibility Notes

The project uses fixed seeds and stores split metadata so experiments can be resumed consistently. Reproducibility is strengthened by:

- patient-level splitting,
- saved client partitions,
- saved scanner profiles,
- saved RNG states in round checkpoints.

## Included Results Directory

The repository is configured to keep most of `results/` versioned, including lightweight plots, CSV files, and JSON summaries.

The large binary file `results/data_splits.pt` is intentionally excluded from git to keep pushes manageable.

## Typical Workflow

```bash
cd /home/khalifa.juma/HC701/Project
conda env create -f environment.yml
conda activate fl_cxr

# Quick validation
python benchmark.py --algo FedAvg --alpha 1.0

# Or full sweep
python benchmark.py

# Generate additional analysis plots
python generate_fairness_plots.py
python generate_roc_plot.py
```

## Future Improvements

Useful next steps for the project could include:

- resolving the Opacus/ViT compatibility issue so `DP-FedAvg` runs end-to-end,
- adding a non-ViT backbone for broader comparisons,
- exporting experiment configs to a dedicated manifest,
- adding automated tests for data loading and checkpoint resume behavior,
- parameterizing dataset paths through environment variables instead of hard-coded local paths.

## Author Notes

This repository is structured for experimentation on a local workstation or SLURM-managed HPC cluster. If you clone it on another machine, the first thing to update is the dataset path configuration in `config.py`.
