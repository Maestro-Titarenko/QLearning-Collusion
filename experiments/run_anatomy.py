"""
Reproduces the paper's Section IV "anatomy of collusion": trains a batch of
sessions, and for each session's converged limit cycle, runs the exogenous
deviation impulse-response analysis at every state and for both deviator
identities, then aggregates the way the paper does (first averaging over
"each starting point on the cycle" within a session to get one observation
for that session, then averaging across sessions).

Usage:
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

    # Within a session, average over each starting point on the cycle x both
    # deviator identities to get one observation for this session (matching
    # the paper's footnote 29 treatment)
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
    print(f"Training and analyzing {len(seeds)} session(s) (deviation/impulse-response analysis), seed={seeds[0]}..{seeds[-1]}")
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_train_and_analyze)(seed, params, pmat_full, pmat_flat, piN, piM, strides) for seed in seeds
    )
    results = [r for r in results if r is not None]
    print(f"Done, elapsed {time.time()-t0:.1f}s, {len(results)}/{len(seeds)} session(s) converged")

    payload = {
        "alpha": ALPHA, "beta": BETA, "horizon": HORIZON,
        "grid0": grids[0].tolist(),
        "p_nash": p_nash.tolist(), "p_monopoly": p_monopoly.tolist(),
        "sessions": results,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f)
    print(f"Results written to {RESULTS_PATH}")

    all_pct_gains = np.array([g for r in results for g in r["pct_gains"]])
    all_profitable = np.array([g for r in results for g in r["profitable_flags"]])
    print(f"\nMean discounted-profit change from deviating: {all_pct_gains.mean()*100:.2f}% (paper's Table 3: about -3% to -4%)")
    print(f"Share of deviations that are unprofitable: {(~all_profitable).mean()*100:.1f}% (paper: >95%)")


if __name__ == "__main__":
    main()
