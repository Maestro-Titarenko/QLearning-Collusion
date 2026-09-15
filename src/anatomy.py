"""
Paper Section IV, "Anatomy of Collusion": deviation-punishment analysis.

Core operation (matching Figure 4/5 and Table 2/3 in the paper): starting
from some state s0 on the converged limit cycle, exogenously force one player
to deviate to the "static best response" at τ=1 (i.e. assuming the rival
still prices at what s0 implies, the deviator picks the price that maximizes
its own per-period profit), then from τ=2 onward both players resume their
learned (fixed, no longer exploring) policies, and observe how prices evolve
and whether the deviation pays off in discounted profit.

The key simplification here matches simulate.py: once the policy is fixed,
the whole system is a deterministic finite automaton (finitely many states),
so any price path rolled forward from a given point under the fixed policy
is guaranteed to eventually enter some cycle (pigeonhole principle). This
lets us compute the infinite-horizon discounted profit exactly (not by
truncation), with no numerical truncation error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def decode_state(state: int, n: int, m: int, strides: np.ndarray) -> np.ndarray:
    """Decode a flattened integer state back into each player's price index
    (strides follow the same convention as simulate._encode_strides: player
    0 is the most significant digit)."""
    actions = np.empty(n, dtype=int)
    remaining = state
    for idx in range(n):
        actions[idx] = remaining // strides[idx]
        remaining = remaining % strides[idx]
    return actions


def static_best_response_2p(profit_matrix_full: np.ndarray, deviating_player: int, rival_action: int) -> int:
    """For the duopoly (n=2) case, given the rival's price index, the
    deviator's static-best-response price index. profit_matrix_full is the
    raw (unflattened) output of build_profit_matrix(), shape (m, m, 2)."""
    if deviating_player == 0:
        profits_given_a0 = profit_matrix_full[:, rival_action, 0]
    elif deviating_player == 1:
        profits_given_a0 = profit_matrix_full[rival_action, :, 1]
    else:
        raise ValueError("This function currently only supports the n=2 duopoly")
    return int(np.argmax(profits_given_a0))


def _decompose_trajectory(policy: np.ndarray, strides: np.ndarray, first_state: int, max_steps: int) -> Tuple[List[int], List[int]]:
    """Starting from first_state, roll forward deterministically under the
    fixed policy and split the path into a "transient part before entering
    the cycle" and a "cyclic part." Returns (transient_states,
    cycle_states)."""
    visited = {}
    seq: List[int] = []
    state = first_state
    for step in range(max_steps):
        if state in visited:
            cycle_start = visited[state]
            return seq[:cycle_start], seq[cycle_start:]
        visited[state] = step
        seq.append(state)
        state = int(np.dot(policy[:, state], strides))
    raise RuntimeError("Failed to find a limit cycle within max_steps")


def discounted_value_from(
    policy: np.ndarray,
    profit_matrix_flat: np.ndarray,
    strides: np.ndarray,
    delta: float,
    first_state: int,
    max_steps: int,
) -> np.ndarray:
    """Exact computation: starting from first_state (i.e. "the price state
    that actually realizes in period 1"), playing forever under the fixed
    policy, each player's infinite-horizon discounted profit
    sum_{t=0}^inf delta^t * pi(s_t). Because the trajectory is eventually
    periodic, this is split into a "transient + cycle" exact analytic sum,
    with no truncation approximation:
        V = sum_{transient} delta^t * r_t  +  delta^{transient length} * (discount-weighted sum over the cycle) / (1 - delta^L)
    """
    transient, cycle = _decompose_trajectory(policy, strides, first_state, max_steps)
    n = profit_matrix_flat.shape[1]
    V = np.zeros(n)
    for t, s in enumerate(transient):
        V += (delta**t) * profit_matrix_flat[s]
    L = len(cycle)
    cycle_val = np.zeros(n)
    for j, s in enumerate(cycle):
        cycle_val += (delta**j) * profit_matrix_flat[s]
    V += (delta ** len(transient)) * cycle_val / (1.0 - delta**L)
    return V


@dataclass
class DeviationOutcome:
    s0: int
    deviating_player: int
    price_path: np.ndarray  # shape (horizon+1, n); index 0 is τ=0 (before the deviation)
    v_baseline: np.ndarray  # shape (n,); discounted profit from τ=1 onward without deviating
    v_deviation: np.ndarray  # shape (n,); discounted profit from τ=1 onward after deviating
    pct_gain_deviator: float  # percentage change in the deviator's discounted profit (the quantity in the paper's Table 3 Panel A)
    profitable: bool


def analyze_deviation(
    policy: np.ndarray,
    profit_matrix_full: np.ndarray,
    profit_matrix_flat: np.ndarray,
    strides: np.ndarray,
    delta: float,
    s0: int,
    deviating_player: int,
    n: int,
    m: int,
    horizon: int,
    max_steps: int,
) -> DeviationOutcome:
    """For one concrete (starting state s0, deviator) combination, compute
    the price impulse-response path and whether the deviation pays off.
    static_best_response_2p currently only supports n=2, so this function is
    likewise restricted to n=2 (matching the paper's Section IV baseline
    experiment setup)."""
    if n != 2:
        raise NotImplementedError("Only the duopoly (n=2) deviation analysis is implemented so far")

    rival = 1 - deviating_player
    s0_actions = decode_state(s0, n, m, strides)

    # τ=1: the deviator plays the static best response, while the rival
    # still prices according to its normal policy
    dev_action = static_best_response_2p(profit_matrix_full, deviating_player, s0_actions[rival])
    actions_tau1 = policy[:, s0].copy()
    actions_tau1[deviating_player] = dev_action
    s1_deviation = int(np.dot(actions_tau1, strides))

    # No-deviation baseline: both players follow their normal policy from τ=1 onward
    s1_baseline = int(np.dot(policy[:, s0], strides))

    v_baseline = discounted_value_from(policy, profit_matrix_flat, strides, delta, s1_baseline, max_steps)
    v_deviation = discounted_value_from(policy, profit_matrix_flat, strides, delta, s1_deviation, max_steps)

    # Price path (for the impulse-response plot): τ=0 is the pre-deviation
    # state, τ=1 is the actions_tau1 just computed, and from τ=2 onward it
    # continues under the normal policy
    price_path = [s0_actions, actions_tau1]
    state = s1_deviation
    for _ in range(horizon - 1):
        actions = policy[:, state]
        price_path.append(actions.copy())
        state = int(np.dot(actions, strides))
    price_path = np.array(price_path)

    pct_gain = (v_deviation[deviating_player] - v_baseline[deviating_player]) / v_baseline[deviating_player]

    return DeviationOutcome(
        s0=s0,
        deviating_player=deviating_player,
        price_path=price_path,
        v_baseline=v_baseline,
        v_deviation=v_deviation,
        pct_gain_deviator=float(pct_gain),
        profitable=bool(pct_gain > 0),
    )
