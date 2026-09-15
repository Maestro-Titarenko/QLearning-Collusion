"""
Reproduces the paper's Table 4 (cost asymmetry). c1=1 fixed, c2 varies over
the values in the paper's table header, with a_1=a_2=2 held constant
(quality symmetric, purely a cost asymmetry — this is the actual Table 4
setup in the paper). Uses the same representative alpha/beta as Table 4.

**A pitfall hit during development**: the first version of the code, trying
to "preserve a_i-c_i=1," also adjusted a_2 along with c2 (a_2=c2+1), which
made the whole game mathematically equivalent to the c2=1 baseline (just a
uniform price shift), leaving firm 2's static Nash market share stuck at
exactly 0.5 — completely inconsistent with the paper's Table 4, where the
share rises from 0.5 all the way to 0.722 as c2 falls. This turned out to be
a good "sanity check" in its own right: if the market share doesn't move at
all with the degree of asymmetry, the asymmetry wasn't actually introduced.
After fixing a to be constant and varying only cost, the market share began
to move correctly with c2.
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
C2_VALUES = [1.0, 0.875, 0.75, 0.625, 0.5, 0.25]  # the paper's Table 4 header values

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
        print(f"Resuming: already have results for {list(all_results.keys())}, skipping these c2 values")

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

        print(f"c2={c2}: p_nash={p_nash}, p_monopoly={p_monopoly}, firm 2's Nash market share={share2:.3f}")

        t0 = time.time()
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(_run_one)(seed, params, profit_matrix_flat, float(pi_nash_vec.mean()), float(pi_monopoly_vec.mean()))
            for seed in range(args.n_sessions)
        )
        print(f"  {args.n_sessions} session(s) took {time.time()-t0:.1f}s")

        all_results[str(c2)] = {
            "p_nash": p_nash.tolist(),
            "p_monopoly": p_monopoly.tolist(),
            "pi_nash": pi_nash_vec.tolist(),
            "pi_monopoly": pi_monopoly_vec.tolist(),
            "share2_nash": share2,
            "sessions": results,
        }
        # Save to disk after each c2 finishes, so a timeout on a later call
        # doesn't lose results already computed
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f)
        print(f"  Saved results for c2={c2} to {RESULTS_PATH}")

    print("\n=== Summary (compared against the paper's Table 4) ===")
    print(f"{'c2':>6} {'share2':>8} {'Delta':>8} {'pi1/pi1^N':>10} {'pi2/pi2^N':>10}")
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
