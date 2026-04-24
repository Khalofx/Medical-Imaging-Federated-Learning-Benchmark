#!/bin/bash
#SBATCH --job-name=fl_cxr_smoke              # Job name
#SBATCH --output=logs_slurm/%x_%j.out        # Output log file
#SBATCH --error=logs_slurm/%x_%j.err         # Error log file
#SBATCH --partition=ws-ia                    # Partition
#SBATCH --time=24:00:00                      
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G

set -eo pipefail

# ── Project paths ─────────────────────────────────────────────
PROJECT_DIR="/home/khalifa.juma/HC701/Project"
CONDA_ENV="fl_cxr"

mkdir -p "$PROJECT_DIR/logs_slurm"

echo "============================================"
echo "SMOKE TEST"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Start time   : $(date)"
echo "============================================"

# ── Activate conda ────────────────────────────────────────────
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

echo "Python       : $(which python)"
echo "Conda env    : $CONDA_DEFAULT_ENV"

cd "$PROJECT_DIR"
export NUM_WORKERS=$SLURM_CPUS_PER_TASK

# ── Temporarily override config for a minimal run ────────────
# 1 algorithm, 1 α, 2 rounds, 1 local epoch
export FL_SMOKE_TEST=1

echo "Running smoke test: FedAvg, α=1.0, 2 rounds, 1 epoch..."
python benchmark.py --algo FedAvg --alpha 1.0

echo "============================================"
echo "Smoke test finished at: $(date)"
echo "============================================"
