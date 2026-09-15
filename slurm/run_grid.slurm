#!/bin/bash
# (alpha, beta) grid sweep reproducing the paper's Figure 1/2 (CLAUDE.md
# Section 13's "Phase 5," which had been on hold because the local sandbox
# didn't have enough compute).
#
# One array task handles one alpha value, running n_beta * n_sessions
# sessions for that alpha; within a task, joblib uses however many cores
# --cpus-per-task requests.
#
# Usage:
#   cd ~/qlearning-collusion
#   mkdir -p slurm_logs
#   sbatch --array=0-9 slurm/run_grid.slurm          # 10x10 grid, 20 sessions per cell
#
# Resumable: if an array task doesn't finish (killed on timeout),
# resubmitting the same --array skips (beta_idx, seed) combinations already
# written to results/grid/alpha_XXX.jsonl.
#
# To change the grid size: edit N_ALPHA/N_BETA/N_SESSIONS below, and update
# sbatch's --array upper bound to N_ALPHA-1 to match (the two must stay
# consistent — the number of array tasks = N_ALPHA).

#SBATCH --job-name=qcollusion-grid
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --account=rc_classwork_pi
#SBATCH --output=slurm_logs/grid_%A_%a.out
#SBATCH --error=slurm_logs/grid_%A_%a.err

set -euo pipefail

N_ALPHA=10
N_BETA=10
N_SESSIONS=20
MAX_PERIODS=10000000
CONV_THRESHOLD=100000

module purge
module load python/3.12.4
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

cd "${SLURM_SUBMIT_DIR}"
python3 experiments/run_grid.py \
    --alpha_idx "${SLURM_ARRAY_TASK_ID}" \
    --n_alpha "${N_ALPHA}" --n_beta "${N_BETA}" --n_sessions "${N_SESSIONS}" \
    --max_periods "${MAX_PERIODS}" --conv_threshold "${CONV_THRESHOLD}" \
    --n_jobs "${SLURM_CPUS_PER_TASK}"
