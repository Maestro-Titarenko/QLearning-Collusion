"""
Reproduces the paper's Section V.A (robustness to the number of firms):
n=3, n=4, still using the same representative alpha=0.15, beta=4e-6.
Resumable (results for each n go to a separate file).

Usage:
    python3 experiments/run_n_players.py --n 3 --n_sessions 20
    python3 experiments/run_n_players.py --n 4 --n_sessions 10
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

ALPHA = 0.15
BETA = 4e-6
CONV_THRESHOLD = 100_000


def _run_one(seed: int, params, profit_matrix_flat, pi_nash, pi_monopoly, max_periods) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    res = run_session(
        params, profit_matrix_flat, pi_nash, pi_monopoly,
        alpha=ALPHA, beta=BETA, max_periods=max_periods, conv_threshold=CONV_THRESHOLD, rng=rng, chunk_size=300_000,
    )
    return {"seed": seed, "converged": res.converged, "n_periods": res.n_periods, "delta": res.delta, "elapsed_sec": time.time() - t0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--n_sessions", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_periods", type=int, default=15_000_000)
    parser.add_argument("--n_jobs", type=int, default=2)
    args = parser.parse_args()

    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", f"n{args.n}_experiment.jsonl")

    params = EconParams.baseline(n=args.n)
    p_nash, p_monopoly = solve_nash(params), solve_monopoly(params)
    grids = build_price_grid(params, p_nash, p_monopoly)
    profit_matrix_flat = build_profit_matrix(params, grids).reshape(-1, params.n)
    pi_nash = float(profits(p_nash, params.c, params.a, params.a0, params.mu).mean())
    pi_monopoly = float(profits(p_monopoly, params.c, params.a, params.a0, params.mu).mean())

    done_seeds = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    done_seeds.add(json.loads(line)["seed"])

    todo = [s for s in range(args.start, args.start + args.n_sessions) if s not in done_seeds]
    if not todo:
        print("All requested seeds have already been run")
        return

    print(f"n={args.n}, state space size S={params.m**params.n}, running {len(todo)} session(s) (seed={todo[0]}..{todo[-1]})")
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_run_one)(seed, params, profit_matrix_flat, pi_nash, pi_monopoly, args.max_periods) for seed in todo
    )
    print(f"Done, elapsed {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    all_rows = [json.loads(l) for l in open(results_path)]
    deltas = np.array([r["delta"] for r in all_rows if r["converged"]])
    conv_rate = np.mean([r["converged"] for r in all_rows])
    print(f"\n{len(all_rows)} session(s) total: convergence rate={conv_rate:.2f}, mean Delta over converged sessions={deltas.mean():.3f} (std {deltas.std():.3f})")


if __name__ == "__main__":
    main()
