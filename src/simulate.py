"""
Session driver for the duopoly repeated pricing game. Corresponds to the
paper's Section II.E-III.B.

Performance note: this sandbox can't install numba, so this is a hand-tuned,
pure-NumPy/Python implementation optimized for the no-JIT case:
  1. States are represented as flattened integer indices (rather than
     tuples), and the Q matrix is reshaped to (n, S, m).
  2. The profit matrix is pre-flattened to (S, n): because under k=1 memory
     the encoding used for "next state" and the encoding used for "this
     period's action combination" are the same mapping, the reward is
     simply profit_matrix_flat[next_state] with no extra action-encoding
     step needed.
  3. Each chunk (e.g. 200,000 periods) pre-generates random numbers in bulk
     (whether to explore, and which action to pick when exploring at
     random), instead of calling the random number generator once per
     period.
  4. The convergence criterion is optimized to an equivalent but cheaper
     check. The paper's criterion is "the optimal action is unchanged for N
     consecutive periods, for every player and every state." Since
     Q-learning only ever updates the one (state, action) cell that was
     visited each period, no other cell's Q-value — and hence no other
     state's greedy action — can possibly change. So it's enough to check,
     after each update, "did the greedy action of the state that was just
     updated change this period?" If there's no such change for N
     consecutive periods, that is equivalent to the whole policy having been
     unchanged for N consecutive periods. This is much faster than
     recomputing the full argmax(Q, axis=-1) every period.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.environment import EconParams, build_price_grid, build_profit_matrix, profits, solve_monopoly, solve_nash
from src.qlearning import init_q_matrix


@dataclass
class SessionResult:
    converged: bool
    n_periods: int
    cycle_states: List[int]
    cycle_length: int
    avg_profit: np.ndarray  # shape (n,), long-run (post-convergence) average per-period profit
    delta: float  # eq. (9) from the paper, computed from the average profit across all players
    trace_periods: List[int] = field(default_factory=list)
    trace_avg_profit: List[float] = field(default_factory=list)
    policy: np.ndarray | None = None  # shape (n, S), the converged greedy limit policy;
    # kept by default (the array is small — only 2x225 ints at n=2) for reuse
    # by the Section IV "anatomy of collusion" deviation/impulse-response
    # analysis, which would otherwise need to retrain. Exclude this field
    # manually when serializing to jsonl.


@dataclass
class ExperimentResult:
    sessions: List[SessionResult] = field(default_factory=list)

    @property
    def delta_array(self) -> np.ndarray:
        return np.array([s.delta for s in self.sessions])

    @property
    def converged_frac(self) -> float:
        return float(np.mean([s.converged for s in self.sessions]))

    @property
    def cycle_length_counts(self) -> dict:
        out: dict = {}
        for s in self.sessions:
            out[s.cycle_length] = out.get(s.cycle_length, 0) + 1
        return out


def _encode_strides(n: int, m: int) -> np.ndarray:
    """Aligned with the flattening order of meshgrid(indexing='ij') in
    build_profit_matrix(): player 0's action is the most significant
    digit."""
    return m ** np.arange(n - 1, -1, -1)


def _extract_steady_state(policy: np.ndarray, profit_matrix_flat: np.ndarray, s_start: int, strides: np.ndarray, max_steps: int) -> tuple[List[int], np.ndarray]:
    """Starting from a given state, deterministically roll forward under the
    fixed (greedy) policy until a state repeats, thereby exactly identifying
    the limit cycle (which may have length 1 — a constant price — or be a
    longer price cycle). In a finite state space, max_steps = S+1 is always
    enough to guarantee a repeat occurs (pigeonhole principle).
    """
    visited = {}
    trajectory: List[int] = []
    state = s_start
    for step in range(max_steps):
        if state in visited:
            cycle_start = visited[state]
            cycle_states = trajectory[cycle_start:]
            avg_profit = profit_matrix_flat[cycle_states].mean(axis=0)
            return cycle_states, avg_profit
        visited[state] = step
        trajectory.append(state)
        actions = policy[:, state]
        state = int(np.dot(actions, strides))
    # Should be unreachable in theory (a finite state space always produces a
    # repeat within S+1 steps).
    raise RuntimeError("Failed to find a limit cycle within max_steps; check that the state-space size is set correctly")


def run_session(
    params: EconParams,
    profit_matrix_flat: np.ndarray,
    pi_nash: float,
    pi_monopoly: float,
    alpha: float,
    beta: float,
    max_periods: int,
    conv_threshold: int,
    rng: np.random.Generator,
    chunk_size: int = 200_000,
    record_every: int | None = None,
) -> SessionResult:
    """When record_every is not None, additionally records a windowed
    average-profit trace during training (in the returned
    SessionResult.trace fields), used to plot a learning curve similar to
    the paper's Figure 10. This recording has a small overhead of its own,
    so it's off by default and only turned on when running a single example
    session meant for plotting."""
    n, m = params.n, params.m
    S = m**n
    strides = _encode_strides(n, m)

    Q = init_q_matrix(profit_matrix_flat.reshape((m,) * n + (n,)), params.delta, n, m, params.k).reshape(n, S, m)
    policy = np.argmax(Q, axis=-1)  # shape (n, S)

    state = int(rng.integers(0, S))
    periods_since_change = 0
    t = 0
    converged = False

    trace_periods: List[int] = []
    trace_avg_profit: List[float] = []
    window_sum = 0.0
    window_count = 0

    while t < max_periods and not converged:
        this_chunk = min(chunk_size, max_periods - t)
        epsilons = np.exp(-beta * np.arange(t, t + this_chunk))
        explore_draws = rng.random((this_chunk, n))
        explore_mask = explore_draws < epsilons[:, None]
        random_actions = rng.integers(0, m, size=(this_chunk, n))

        for j in range(this_chunk):
            actions = np.where(explore_mask[j], random_actions[j], policy[:, state])
            next_state = int(np.dot(actions, strides))
            rewards = profit_matrix_flat[next_state]

            changed = False
            for i in range(n):
                a_i = int(actions[i])
                best_next = Q[i, next_state, :].max()
                td_target = rewards[i] + params.delta * best_next
                old = Q[i, state, a_i]
                Q[i, state, a_i] = (1.0 - alpha) * old + alpha * td_target
                new_best = int(np.argmax(Q[i, state, :]))
                if new_best != policy[i, state]:
                    policy[i, state] = new_best
                    changed = True

            state = next_state
            t += 1

            if record_every is not None:
                window_sum += float(rewards.mean())
                window_count += 1
                if t % record_every == 0:
                    trace_periods.append(t)
                    trace_avg_profit.append(window_sum / window_count)
                    window_sum = 0.0
                    window_count = 0

            if changed:
                periods_since_change = 0
            else:
                periods_since_change += 1
                if periods_since_change >= conv_threshold:
                    converged = True
                    break

    cycle_states, avg_profit = _extract_steady_state(policy, profit_matrix_flat, state, strides, max_steps=S + 1)
    pi_bar = float(np.mean(avg_profit))
    delta = (pi_bar - pi_nash) / (pi_monopoly - pi_nash)

    return SessionResult(
        converged=converged,
        n_periods=t,
        trace_periods=trace_periods,
        trace_avg_profit=trace_avg_profit,
        cycle_states=cycle_states,
        cycle_length=len(cycle_states),
        avg_profit=avg_profit,
        delta=delta,
        policy=policy.copy(),
    )


def run_experiment(
    params: EconParams,
    alpha: float,
    beta: float,
    n_sessions: int,
    max_periods: int,
    conv_threshold: int = 100_000,
    seed: int = 0,
) -> ExperimentResult:
    """Run n_sessions independent sessions and return the aggregated result.
    Corresponds to the paper's design of "one experiment per parameter set,
    1000 sessions per experiment" (here the number of sessions is
    configurable, to allow validating with a small number of sessions
    first)."""
    p_nash = solve_nash(params)
    p_monopoly = solve_monopoly(params)
    grids = build_price_grid(params, p_nash, p_monopoly)
    profit_matrix = build_profit_matrix(params, grids)
    profit_matrix_flat = profit_matrix.reshape(-1, params.n)

    pi_nash = float(profits(p_nash, params.c, params.a, params.a0, params.mu).mean())
    pi_monopoly = float(profits(p_monopoly, params.c, params.a, params.a0, params.mu).mean())

    result = ExperimentResult()
    for s in range(n_sessions):
        rng = np.random.default_rng(seed + s)
        session = run_session(
            params, profit_matrix_flat, pi_nash, pi_monopoly, alpha, beta, max_periods, conv_threshold, rng
        )
        result.sessions.append(session)
    return result
