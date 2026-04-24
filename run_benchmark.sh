#!/bin/bash
#SBATCH --job-name=fl_cxr_benchmark          # Job name
#SBATCH --output=logs_slurm/%x_%j.out        # Output log file
#SBATCH --error=logs_slurm/%x_%j.err         # Error log file
#SBATCH --partition=ws-ia                    # Partition (same as your previous job)
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --time=24:00:00                      # Time limit
#SBATCH --nodes=1                            # Run on 1 node
#SBATCH --ntasks=1                           # Run as 1 task
#SBATCH --cpus-per-task=20                   # 20 CPUs — used by DataLoader workers
#SBATCH --mem=64G                            # 64GB RAM

set -euo pipefail

# ── Project paths ──────────────────────────────────────────────
PROJECT_DIR="/home/khalifa.juma/HC701/Project"
CONDA_ENV="fl_cxr"
BENCHMARK_ENTRY="$PROJECT_DIR/benchmark.py"
DATASET_DIR="$PROJECT_DIR/NIH_Chest_Xrays"

# ── Create log directory if it doesn't exist ──────────────────
mkdir -p "$PROJECT_DIR/logs_slurm"

# ── Print job info ─────────────────────────────────────────────
echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Job Name     : $SLURM_JOB_NAME"
echo "Node         : $SLURMD_NODENAME"
echo "CPUs         : $SLURM_CPUS_PER_TASK"
echo "Memory       : 64G"
echo "Start time   : $(date)"
echo "============================================"

# ── Load conda and activate environment ───────────────────────
# This works on most HPC clusters — if it fails, ask your sysadmin
# which module to load (e.g. "module load anaconda3/2023.09")
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available in this job environment."
    exit 1
fi

# Source conda so the 'conda activate' command is available in scripts
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

echo "Python       : $(which python)"
echo "Conda env    : $CONDA_DEFAULT_ENV"

# ── Move into project directory ────────────────────────────────
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: project directory not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

if [ ! -f "$BENCHMARK_ENTRY" ]; then
    echo "ERROR: benchmark entrypoint not found: $BENCHMARK_ENTRY"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: dataset directory not found: $DATASET_DIR"
    exit 1
fi

# ── Pass SLURM CPU count to config.py via env variable ────────
export NUM_WORKERS="${SLURM_CPUS_PER_TASK:-4}"

echo "Project dir  : $PROJECT_DIR"
echo "Dataset dir  : $DATASET_DIR"
echo "Workers      : $NUM_WORKERS"

# ── Run benchmark (auto-resumes if job is a restart) ──────────
echo "Starting benchmark..."
python "$BENCHMARK_ENTRY" "$@"

# ── Print finish time ──────────────────────────────────────────
echo "============================================"
echo "Finished at  : $(date)"
echo "============================================"
