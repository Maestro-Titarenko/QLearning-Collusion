"""
The (alpha, beta) grid sweep for the paper's Figure 1/2. CLAUDE.md Section 13
flagged this as "Phase 5, not yet done" — this is that implementation,
designed to be split by SLURM job array: each array task is responsible for
one fixed alpha value, running every beta value x every session for that
alpha, using joblib within the task to fully use a compute node's cores.

The alpha grid range [0.025, 0.25] is the paper's baseline grid range as
recorded in CLAUDE.md Section 4. There's currently no exact paper number on
record in CLAUDE.md for the beta grid range (this repo hadn't done this part
before), so the log-spaced range [1e-6, 2e-5] used here (midpoint close to
1e-5) was back-derived from two known anchor points in CLAUDE.md — "grid
midpoint alpha=0.125, beta=1e-5" (Section 11) and the "representative
experiment point beta=4e-6" (Section 5). **This is an assumption that still
needs checking, not a number taken directly from the paper** — the actual
grid endpoints should be verified against the paper before treating any
resulting figure as final.

Usage (a single array task, corresponding to one alpha value):
    python3 experiments/run_grid.py --alpha_idx 0 --n_alpha 20 --n_beta 20 \
        --n_sessions 20 --max_periods 2000000 --n_jobs 32

Resumable: each alpha_idx has its own output file
results/grid/alpha_{alpha_idx:03d}.jsonl, deduplicated within the file by
(beta_idx, seed).
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
    parser.add_argument("--alpha_idx", type=int, required=True, help="index into the alpha grid that this task is responsible for")
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
    parser.add_argument("--n", type=int, default=2, help="number of firms, defaults to the symmetric duopoly")
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
        print(f"alpha_idx={args.alpha_idx} (alpha={alpha:.4f}) is already fully done, skipping.")
        return

    print(
        f"alpha_idx={args.alpha_idx} (alpha={alpha:.4f}), "
        f"running {len(todo)} (beta, session) combination(s), n_jobs={args.n_jobs}"
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
    print(f"Done, total elapsed {elapsed:.1f}s (average {elapsed/len(todo):.1f}s each)")

    with open(out_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results appended to {out_path}")


if __name__ == "__main__":
    main()
