"""
跑论文的"代表性实验"参数点（alpha=0.15, beta=4e-6，基准对称双寡头），
用来复现 Table 1 的统计量。

用法（可分批跑，断点续跑——已经跑过的 session id 会被跳过）：
    python3 experiments/run_representative.py --n_sessions 50 --start 0
    python3 experiments/run_representative.py --n_sessions 50 --start 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from joblib import Parallel, delayed

from src.environment import EconParams, build_price_grid, build_profit_matrix, profits, solve_monopoly, solve_nash
from src.simulate import run_session

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "representative_experiment.jsonl")

ALPHA = 0.15
BETA = 4e-6
MAX_PERIODS = 4_000_000
CONV_THRESHOLD = 100_000


def _run_one(seed: int, params: EconParams, profit_matrix_flat: np.ndarray, pi_nash: float, pi_monopoly: float) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    res = run_session(
        params, profit_matrix_flat, pi_nash, pi_monopoly,
        alpha=ALPHA, beta=BETA, max_periods=MAX_PERIODS, conv_threshold=CONV_THRESHOLD, rng=rng,
    )
    elapsed = time.time() - t0
    return {
        "seed": seed,
        "converged": res.converged,
        "n_periods": res.n_periods,
        "cycle_length": res.cycle_length,
        "avg_profit": res.avg_profit.tolist(),
        "delta": res.delta,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sessions", type=int, default=50)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=2)
    args = parser.parse_args()

    params = EconParams.baseline(n=2)
    p_nash = solve_nash(params)
    p_monopoly = solve_monopoly(params)
    grids = build_price_grid(params, p_nash, p_monopoly)
    profit_matrix = build_profit_matrix(params, grids)
    profit_matrix_flat = profit_matrix.reshape(-1, params.n)
    pi_nash = float(profits(p_nash, params.c, params.a, params.a0, params.mu).mean())
    pi_monopoly = float(profits(p_monopoly, params.c, params.a, params.a0, params.mu).mean())

    # 断点续跑：已经在结果文件里的 seed 就跳过
    done_seeds = set()
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            for line in f:
                if line.strip():
                    done_seeds.add(json.loads(line)["seed"])

    todo = [s for s in range(args.start, args.start + args.n_sessions) if s not in done_seeds]
    if not todo:
        print(f"seeds {args.start}..{args.start + args.n_sessions - 1} 都已经跑过，跳过。")
        return

    print(f"跑 {len(todo)} 个新 session（seed={todo[0]}..{todo[-1]}），alpha={ALPHA}, beta={BETA}, n_jobs={args.n_jobs}")
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_run_one)(seed, params, profit_matrix_flat, pi_nash, pi_monopoly) for seed in todo
    )
    elapsed = time.time() - t0
    print(f"完成 {len(todo)} 个 session，总耗时 {elapsed:.1f}s（平均每个 {elapsed/len(todo):.1f}s）")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"结果已追加写入 {RESULTS_PATH}")


if __name__ == "__main__":
    main()
