"""
复现论文 Table 4（成本不对称）。c1=1 固定，c2 按论文表头的取值变化，a_1=a_2=2
保持不变（质量对称，纯粹是成本不对称——这是论文 Table 4 的真实设定）。用和
Table 4 一样的代表性 alpha/beta。

**开发时踩过的坑**：第一版代码为了"保持 a_i-c_i=1"把 a_2 也跟着 c2 一起调整
（a_2=c2+1），结果整个博弈在数学上和 c2=1 的基准完全等价（只是价格整体平移），
2 号企业的静态 Nash 市场份额恒等于 0.5，和论文 Table 4 里份额随 c2 下降从 0.5
一路升到 0.722 的模式完全对不上——这时候市场份额这个量本身就是很好的一个
"合理性检验"：如果它没有随不对称程度变化，说明不对称没有真的被引入。改成
a 恒定、只变成本后，市场份额才开始正确地随 c2 变化。
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

from src.environment import EconParams, build_price_grid, build_profit_matrix, demand, profits, solve_monopoly, solve_nash
from src.simulate import run_session

ALPHA = 0.15
BETA = 4e-6
MAX_PERIODS = 4_000_000
CONV_THRESHOLD = 100_000
C2_VALUES = [1.0, 0.875, 0.75, 0.625, 0.5, 0.25]  # 论文 Table 4 的表头

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "asymmetric_experiment.json")


def _run_one(seed: int, params: EconParams, profit_matrix_flat: np.ndarray, pi_nash_mean: float, pi_monopoly_mean: float) -> dict:
    rng = np.random.default_rng(seed)
    res = run_session(
        params, profit_matrix_flat, pi_nash_mean, pi_monopoly_mean,
        alpha=ALPHA, beta=BETA, max_periods=MAX_PERIODS, conv_threshold=CONV_THRESHOLD, rng=rng,
    )
    return {"seed": seed, "converged": res.converged, "delta": res.delta, "avg_profit": res.avg_profit.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sessions", type=int, default=15)
    parser.add_argument("--n_jobs", type=int, default=2)
    args = parser.parse_args()

    all_results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        print(f"断点续跑：已有 {list(all_results.keys())} 的结果，跳过这些 c2")

    for c2 in C2_VALUES:
        if str(c2) in all_results:
            continue
        params = EconParams(n=2, c=np.array([1.0, c2]), a=np.array([2.0, 2.0]), a0=0.0, mu=0.25, delta=0.95, m=15, xi=0.1, k=1)
        p_nash = solve_nash(params)
        p_monopoly = solve_monopoly(params)
        grids = build_price_grid(params, p_nash, p_monopoly)
        profit_matrix = build_profit_matrix(params, grids)
        profit_matrix_flat = profit_matrix.reshape(-1, params.n)

        pi_nash_vec = profits(p_nash, params.c, params.a, params.a0, params.mu)
        pi_monopoly_vec = profits(p_monopoly, params.c, params.a, params.a0, params.mu)
        q_nash = demand(p_nash, params.a, params.a0, params.mu)
        share2 = float(q_nash[1] / (q_nash[0] + q_nash[1]))

        print(f"c2={c2}: p_nash={p_nash}, p_monopoly={p_monopoly}, 2's Nash market share={share2:.3f}")

        t0 = time.time()
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(_run_one)(seed, params, profit_matrix_flat, float(pi_nash_vec.mean()), float(pi_monopoly_vec.mean()))
            for seed in range(args.n_sessions)
        )
        print(f"  {args.n_sessions} 个 session 耗时 {time.time()-t0:.1f}s")

        all_results[str(c2)] = {
            "p_nash": p_nash.tolist(),
            "p_monopoly": p_monopoly.tolist(),
            "pi_nash": pi_nash_vec.tolist(),
            "pi_monopoly": pi_monopoly_vec.tolist(),
            "share2_nash": share2,
            "sessions": results,
        }
        # 每跑完一个 c2 就落盘一次，避免单次调用超时把之前跑完的结果也丢掉
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f)
        print(f"  已保存 c2={c2} 的结果到 {RESULTS_PATH}")

    print("\n=== 汇总（对照论文 Table 4）===")
    print(f"{'c2':>6} {'2份额':>8} {'Delta':>8} {'pi1/pi1^N':>10} {'pi2/pi2^N':>10}")
    for c2 in C2_VALUES:
        r = all_results[str(c2)]
        deltas = np.array([s["delta"] for s in r["sessions"]])
        avg_profits = np.array([s["avg_profit"] for s in r["sessions"]])  # (n_sessions, 2)
        pi_nash_vec = np.array(r["pi_nash"])
        ratio1 = avg_profits[:, 0].mean() / pi_nash_vec[0]
        ratio2 = avg_profits[:, 1].mean() / pi_nash_vec[1]
        print(f"{c2:>6} {r['share2_nash']:>8.3f} {deltas.mean():>8.3f} {ratio1:>10.3f} {ratio2:>10.3f}")


if __name__ == "__main__":
    main()
