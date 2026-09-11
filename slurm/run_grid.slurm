#!/bin/bash
# 复现论文 Figure 1/2 的 (alpha, beta) 网格扫描（CLAUDE.md 第 13 节标注的
# Phase 5，之前因为本地沙箱算力不够一直没做）。
#
# 一个 array task 负责一个 alpha 值，跑完这个 alpha 下 n_beta * n_sessions 个
# session，task 内部用 joblib 把 --cpus-per-task 指定的核心数用满。
#
# 用法：
#   cd ~/qlearning-collusion
#   mkdir -p slurm_logs
#   sbatch --array=0-19 slurm/run_grid.slurm          # 20x20 网格，每点 20 session
#
# 断点续跑：某个 array task 没跑完/超时被杀，重新 sbatch 同样的 --array 会跳过
# 已经写进 results/grid/alpha_XXX.jsonl 的 (beta_idx, seed) 组合。
#
# 调整网格规模：把下面 N_ALPHA/N_BETA/N_SESSIONS 改掉，同时把 sbatch 的
# --array 上限改成 N_ALPHA-1（两处必须一致，array task 数量 = N_ALPHA）。

#SBATCH --job-name=qcollusion-grid
#SBATCH --partition=general
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --account=rc_classwork_pi
#SBATCH --output=slurm_logs/grid_%A_%a.out
#SBATCH --error=slurm_logs/grid_%A_%a.err

set -euo pipefail

N_ALPHA=20
N_BETA=20
N_SESSIONS=20
MAX_PERIODS=2000000
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
