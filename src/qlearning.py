"""
Core Q-learning primitives: Q-matrix initialization, the update rule,
epsilon-greedy exploration.

Corresponds to Calvano et al. (2020) Section I.A (general Q-learning) and
Section II.D-E (the specific setup when applied to the repeated pricing
game). Equation numbers match CLAUDE.md Sections 4-6.

State representation convention: no manual mixed-radix encoding — state is
represented directly as a NumPy multidimensional array indexed by tuples. For
n players, k-period memory, and m discrete prices:
    Q_i has shape (m,)*(n*k) + (m,)
The first n*k axes are the state (past k periods' price indices for all n
players, in a fixed order), and the last axis is the current player's own
action (price index) to choose. This way, indexing Q_i[state_tuple] directly
gives the vector of Q-values for all actions at that state, and argmax/max
are ordinary NumPy operations with no extra encode/decode layer needed.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def epsilon_greedy_rate(t: int, beta: float) -> float:
    """Eq. (7): epsilon_t = exp(-beta * t)."""
    return float(np.exp(-beta * t))


def init_q_matrix(profit_matrix: np.ndarray, delta: float, n: int, m: int, k: int) -> np.ndarray:
    """Eq. (8): Q_{i,0}(s, a_i) = [ expected per-period profit from choosing
    a_i, assuming rivals price uniformly at random ] / (1-delta).

    This initial value doesn't depend on s (it's the same for every state),
    because at t=0 there's no history yet that could distinguish states from
    one another.

    Parameters
    ----------
    profit_matrix : shape (m,)*n + (n,), the output of build_profit_matrix().
    delta, n, m, k : see EconParams.

    Returns
    -------
    An array of shape (n,) + (m,)*(n*k) + (m,).
    """
    q0 = np.empty((n, m))
    for i in range(n):
        other_axes = tuple(ax for ax in range(n) if ax != i)
        avg_profit_given_ai = profit_matrix[..., i].mean(axis=other_axes)  # shape (m,)
        q0[i] = avg_profit_given_ai / (1.0 - delta)

    state_shape = (m,) * (n * k)
    Q = np.empty((n,) + state_shape + (m,))
    for i in range(n):
        # Broadcasting: q0[i] has shape (m,), aligning with Q[i]'s last
        # (action) axis and automatically filling every state axis.
        Q[i, ...] = q0[i]
    return Q


def greedy_action(q_values: np.ndarray) -> int:
    """Pick the best action from the Q-value vector at a given state. On
    ties, np.argmax defaults to the first (= lowest-index) match, and since
    our price grid is strictly increasing, this is exactly the paper's
    tie-break rule — pick the lowest price on ties."""
    return int(np.argmax(q_values))


def choose_action(Q_i: np.ndarray, state: Tuple[int, ...], t: int, beta: float, m: int, rng: np.random.Generator) -> int:
    """Epsilon-greedy: explore uniformly at random with probability
    epsilon_t, otherwise act greedily."""
    eps = epsilon_greedy_rate(t, beta)
    if rng.random() < eps:
        return int(rng.integers(0, m))
    return greedy_action(Q_i[state])


def update_q(Q_i: np.ndarray, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...], alpha: float, delta: float) -> None:
    """Eq. (4): update, in place, the single (state, action) cell of Q_i that
    was just visited."""
    idx = state + (action,)
    best_next = np.max(Q_i[next_state])
    td_target = reward + delta * best_next
    Q_i[idx] = (1.0 - alpha) * Q_i[idx] + alpha * td_target


def greedy_policy(Q_i: np.ndarray) -> np.ndarray:
    """Return the greedy action for every state, shape = Q_i.shape[:-1]
    (i.e. the state axes only). Used for the convergence criterion: check
    whether this policy has stayed unchanged over many consecutive
    periods."""
    return np.argmax(Q_i, axis=-1)


# ---------------------------------------------------------------------------
# Generic single-agent Q-learning trainer: fully decoupled from the specific
# economic environment, depending only on a state-transition function
# step(state, action) -> (reward, next_state). Used for:
#   1) verifying correctness on a small MDP with a known closed-form solution
#      in tests/test_qlearning_sanity.py
#   2) reuse by any future single-agent benchmark test
# The repeated-game (multi-agent, simultaneous learning) driver logic lives
# in simulate.py instead, because there each period has to advance two (or
# more) agents together and handle the joint state transition, which has a
# different loop structure from the single-agent case here.
# ---------------------------------------------------------------------------
StepFn = Callable[[Tuple[int, ...], int, np.random.Generator], Tuple[float, Tuple[int, ...]]]


def train_single_agent(
    step_fn: StepFn,
    state_shape: Tuple[int, ...],
    n_actions: int,
    alpha: float,
    beta: float,
    delta: float,
    n_periods: int,
    rng: np.random.Generator,
    initial_state: Tuple[int, ...],
    q_init_value: float = 0.0,
) -> np.ndarray:
    """Run a single-agent, finite-horizon Q-learning training loop and return
    the final Q matrix.

    step_fn(state, action, rng) -> (reward, next_state) encapsulates the
    environment dynamics; the trainer itself doesn't care what that
    environment is (a stateless bandit, or a multi-state MDP, both work).
    """
    Q = np.full(state_shape + (n_actions,), q_init_value, dtype=float)
    state = initial_state
    for t in range(n_periods):
        action = choose_action(Q, state, t, beta, n_actions, rng)
        reward, next_state = step_fn(state, action, rng)
        update_q(Q, state, action, reward, next_state, alpha, delta)
        state = next_state
    return Q
