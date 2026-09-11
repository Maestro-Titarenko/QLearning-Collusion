"""
论文第四节"合谋解剖"的复现：训练一批 session，对每个 session 收敛后的极限环
上每一个状态、两种偏离方身份，做外生偏离的脉冲响应分析，然后按论文的口径
聚合（session 内先对"循环上的各个起点"取平均，算作这个 session 的一个观测，
再对 session 取平均）。

用法：
    python3 experiments/run_anatomy.py --n_sessions 30 --seed_start 0
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

from src.anatomy import analyze_deviation
from src.environment import EconParams, build_price_grid, build_profit_matrix, profits, solve_monopoly, solve_nash
from src.simulate import _encode_strides, run_session

ALPHA = 0.15
BETA = 4e-6
MAX_PERIODS = 4_000_000
CONV_THRESHOLD = 100_000
HORIZON = 15

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "anatomy_experiment.json")


def _train_and_analyze(seed: int, params, pmat_full, pmat_flat, piN, piM, strides):
    rng = np.random.default_rng(seed)
    session = run_session(
        params, pmat_flat, piN, piM, alpha=ALPHA, beta=BETA,
        max_periods=MAX_PERIODS, conv_threshold=CONV_THRESHOLD, rng=rng,
    )
    if not session.converged:
        return None

    # session 内，对循环上的每个起点 x 两种偏离方身份取平均，算这个 session
    # 的一个观测（跟论文脚注 29 的处理方式一致）
    deviator_paths = []
    nondeviator_paths = []
    pct_gains = []
    profitable_flags = []

    for s0 in session.cycle_states:
        for dev_player in (0, 1):
            outcome = analyze_deviation(
                session.policy, pmat_full, pmat_flat, strides, params.delta,
                s0, dev_player, n=params.n, m=params.m, horizon=HORIZON, max_steps=params.m ** params.n + 1,
            )
            nondev_player = 1 - dev_player
            deviator_paths.append(outcome.price_path[:, dev_player])
            nondeviator_paths.append(outcome.price_path[:, nondev_player])
            pct_gains.append(outcome.pct_gain_deviator)
            profitable_flags.append(outcome.profitable)

    session_deviator_path = np.mean(deviator_paths, axis=0)
    session_nondeviator_path = np.mean(nondeviator_paths, axis=0)

    return {
        "seed": seed,
        "delta": session.delta,
        "cycle_length": session.cycle_length,
        "deviator_path": session_deviator_path.tolist(),
        "nondeviator_path": session_nondeviator_path.tolist(),
        "pct_gains": pct_gains,
        "profitable_flags": profitable_flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sessions", type=int, default=30)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=2)
    args = parser.parse_args()

    params = EconParams.baseline(n=2)
    p_nash, p_monopoly = solve_nash(params), solve_monopoly(params)
    grids = build_price_grid(params, p_nash, p_monopoly)
    pmat_full = build_profit_matrix(params, grids)
    pmat_flat = pmat_full.reshape(-1, params.n)
    piN = float(profits(p_nash, params.c, params.a, params.a0, params.mu).mean())
    piM = float(profits(p_monopoly, params.c, params.a, params.a0, params.mu).mean())
    strides = _encode_strides(params.n, params.m)

    seeds = list(range(args.seed_start, args.seed_start + args.n_sessions))
    print(f"训练并分析 {len(seeds)} 个 session（偏离/脉冲响应分析），seed={seeds[0]}..{seeds[-1]}")
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_train_and_analyze)(seed, params, pmat_full, pmat_flat, piN, piM, strides) for seed in seeds
    )
    results = [r for r in results if r is not None]
    print(f"完成，耗时 {time.time()-t0:.1f}s，{len(results)}/{len(seeds)} 个 session 收敛")

    payload = {
        "alpha": ALPHA, "beta": BETA, "horizon": HORIZON,
        "grid0": grids[0].tolist(),
        "p_nash": p_nash.tolist(), "p_monopoly": p_monopoly.tolist(),
        "sessions": results,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f)
    print(f"结果写入 {RESULTS_PATH}")

    all_pct_gains = np.array([g for r in results for g in r["pct_gains"]])
    all_profitable = np.array([g for r in results for g in r["profitable_flags"]])
    print(f"\n偏离的平均贴现利润变化: {all_pct_gains.mean()*100:.2f}% (论文 Table 3 大约 -3% 到 -4%)")
    print(f"偏离不划算（unprofitable）的比例: {(~all_profitable).mean()*100:.1f}% (论文: >95%)")


if __name__ == "__main__":
    main()
