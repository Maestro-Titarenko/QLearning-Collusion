"""
论文 Figure 1/2 的 (alpha, beta) 网格扫描。CLAUDE.md 第 13 节标注为"Phase 5，
还没做的部分"——这是那部分的实现，设计成可以按 SLURM job array 切分：每个
array task 负责固定一个 alpha 值，跑完这个 alpha 下所有 beta 值 x 所有
session，任务内部用 joblib 把一个计算节点的核心全部用满。

alpha 网格范围 [0.025, 0.25] 是 CLAUDE.md 第 4 节写明的论文基准网格范围。beta
网格范围目前没有 CLAUDE.md 记录的确切论文数字（本仓库之前也没做过这部分），
这里按 CLAUDE.md 里两个已知锚点——"网格中点 alpha=0.125, beta=1e-5"（第 11
节）和"代表性实验点 beta=4e-6"（第 5 节）——取了一个对数等距的合理区间
[1e-6, 2e-5]，中点接近 1e-5。**这是一个待核对的假设，不是从论文原文抄的数
字**，正式出图前应该找论文原文核实实际网格端点。

用法（单个 array task，对应一个 alpha 值）：
    python3 experiments/run_grid.py --alpha_idx 0 --n_alpha 20 --n_beta 20 \
        --n_sessions 20 --max_periods 2000000 --n_jobs 32

断点续跑：每个 (alpha_idx) 对应一个独立输出文件
results/grid/alpha_{alpha_idx:03d}.jsonl，文件内按 (beta_idx, seed) 去重。
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

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "grid")


def _run_one(
    alpha: float,
    beta: float,
    beta_idx: int,
    seed: int,
    params: EconParams,
    profit_matrix_flat: np.ndarray,
    pi_nash: float,
    pi_monopoly: float,
    max_periods: int,
    conv_threshold: int,
) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    res = run_session(
        params, profit_matrix_flat, pi_nash, pi_monopoly,
        alpha=alpha, beta=beta, max_periods=max_periods, conv_threshold=conv_threshold, rng=rng,
    )
    elapsed = time.time() - t0
    return {
        "alpha": alpha,
        "beta": beta,
        "beta_idx": beta_idx,
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
    parser.add_argument("--alpha_idx", type=int, required=True, help="这个 task 负责的 alpha 在网格里的下标")
    parser.add_argument("--n_alpha", type=int, default=20)
    parser.add_argument("--n_beta", type=int, default=20)
    parser.add_argument("--alpha_min", type=float, default=0.025)
    parser.add_argument("--alpha_max", type=float, default=0.25)
    parser.add_argument("--beta_min", type=float, default=1e-6)
    parser.add_argument("--beta_max", type=float, default=2e-5)
    parser.add_argument("--n_sessions", type=int, default=20)
    parser.add_argument("--max_periods", type=int, default=2_000_000)
    parser.add_argument("--conv_threshold", type=int, default=100_000)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--n", type=int, default=2, help="企业数量，默认对称双寡头")
    args = parser.parse_args()

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.n_alpha)
    betas = np.geomspace(args.beta_min, args.beta_max, args.n_beta)
    alpha = float(alphas[args.alpha_idx])

    params = EconParams.baseline(n=args.n)
    p_nash = solve_nash(params)
    p_monopoly = solve_monopoly(params)
    grids = build_price_grid(params, p_nash, p_monopoly)
    profit_matrix = build_profit_matrix(params, grids)
    profit_matrix_flat = profit_matrix.reshape(-1, params.n)
    pi_nash = float(profits(p_nash, params.c, params.a, params.a0, params.mu).mean())
    pi_monopoly = float(profits(p_monopoly, params.c, params.a, params.a0, params.mu).mean())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"alpha_{args.alpha_idx:03d}.jsonl")

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["beta_idx"], r["seed"]))

    todo = [
        (beta_idx, seed)
        for beta_idx in range(args.n_beta)
        for seed in range(args.n_sessions)
        if (beta_idx, seed) not in done
    ]
    if not todo:
        print(f"alpha_idx={args.alpha_idx}（alpha={alpha:.4f}）已经全部跑完，跳过。")
        return

    print(
        f"alpha_idx={args.alpha_idx}（alpha={alpha:.4f}），"
        f"跑 {len(todo)} 个 (beta, session) 组合，n_jobs={args.n_jobs}"
    )
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_run_one)(
            alpha, float(betas[beta_idx]), beta_idx, seed,
            params, profit_matrix_flat, pi_nash, pi_monopoly, args.max_periods, args.conv_threshold,
        )
        for beta_idx, seed in todo
    )
    elapsed = time.time() - t0
    print(f"完成，总耗时 {elapsed:.1f}s（平均每个 {elapsed/len(todo):.1f}s）")

    with open(out_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"结果已追加写入 {out_path}")


if __name__ == "__main__":
    main()
