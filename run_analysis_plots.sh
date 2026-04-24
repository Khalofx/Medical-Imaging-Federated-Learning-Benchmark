#!/bin/bash
#SBATCH --job-name=fl_cxr_analysis
#SBATCH --output=logs_slurm/%x_%j.out
#SBATCH --error=logs_slurm/%x_%j.err
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

PROJECT_DIR="/home/khalifa.juma/HC701/Project"
CONDA_ENV="fl_cxr"
ROC_SCRIPT="$PROJECT_DIR/generate_roc_plot.py"
FAIRNESS_SCRIPT="$PROJECT_DIR/generate_fairness_plots.py"
RESULTS_JSON="$PROJECT_DIR/results/completed_experiments.json"
SPLITS_PATH="$PROJECT_DIR/results/data_splits.pt"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
OUT_DIR="$PROJECT_DIR/results"

mkdir -p "$PROJECT_DIR/logs_slurm"

echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Job Name     : $SLURM_JOB_NAME"
echo "Node         : $SLURMD_NODENAME"
echo "CPUs         : $SLURM_CPUS_PER_TASK"
echo "Memory       : 32G"
echo "Start time   : $(date)"
echo "============================================"

module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
    if [ -x "/home/khalifa.juma/miniconda3/bin/conda" ]; then
        export PATH="/home/khalifa.juma/miniconda3/bin:$PATH"
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available in this job environment."
    echo "Tried module load and fallback path: /home/khalifa.juma/miniconda3/bin/conda"
    exit 1
fi

set +u
if [ -f "/home/khalifa.juma/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/khalifa.juma/miniconda3/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"
set -u

echo "Python       : $(which python)"
echo "Conda env    : $CONDA_DEFAULT_ENV"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: project directory not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

for f in "$ROC_SCRIPT" "$FAIRNESS_SCRIPT" "$RESULTS_JSON" "$SPLITS_PATH"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file not found: $f"
        exit 1
    fi
done

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

export NUM_WORKERS="${SLURM_CPUS_PER_TASK:-4}"

DEVICE="cpu"
if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    DEVICE="cuda"
fi

echo "Project dir  : $PROJECT_DIR"
echo "Results file : $RESULTS_JSON"
echo "Checkpoints  : $CHECKPOINT_DIR"
echo "Workers      : $NUM_WORKERS"
echo "Device       : $DEVICE"

echo "Starting fairness plots..."
python "$FAIRNESS_SCRIPT" \
    --device "$DEVICE" \
    --num-workers "$NUM_WORKERS" \
    --results-json "$RESULTS_JSON" \
    --splits-path "$SPLITS_PATH" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --out-dir "$OUT_DIR"

echo "Starting ROC plots..."
python "$ROC_SCRIPT" \
    --device "$DEVICE" \
    --num-workers "$NUM_WORKERS" \
    --results-json "$RESULTS_JSON" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --out-dir "$OUT_DIR"

echo "============================================"
echo "Finished at  : $(date)"
echo "============================================"
