"""
Economic environment: logit demand, profits, static Bertrand-Nash / monopoly
price solvers, action-space discretization.

Corresponds to Calvano et al. (2020) Section II.A-II.E. Equation numbers match
CLAUDE.md Sections 1-2.

FOC derivation (for code review reference; the derivation itself lives in the
project notes, only the conclusions are given here):

    q_i(p) = exp((a_i - p_i)/mu) / [ sum_j exp((a_j - p_j)/mu) + exp(a0/mu) ]

    dq_i/dp_i = -(1/mu) * q_i * (1 - q_i)
    dq_j/dp_i = (1/mu) * q_i * q_j        (j != i)

    Bertrand-Nash (each firm controls only its own price) first-order condition:
        FOC_i = q_i - (1/mu) * q_i * (1-q_i) * (p_i - c_i) = 0

    Monopoly / joint profit maximization (a hypothetical multi-product
    monopolist controlling all prices simultaneously), first-order condition
    after simplification:
        FOC_i = q_i * (1 + (S - (p_i - c_i)) / mu) = 0,   S = sum_j (p_j - c_j) * q_j

    In the symmetric case (n identical products) these reduce to scalar
    equations:
        Nash:     p - c = mu / (1 - q(p))
        Monopoly: p - c = mu / (1 - n * q(p))
    These two scalar equations are used as a cross-check independent of
    fsolve (see tests/test_environment.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize


@dataclass
class EconParams:
    """A set of economic environment parameters. Defaults are the paper's
    baseline (Section II.E)."""

    n: int = 2
    c: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    a: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0]))  # a_i - c_i = 1 => a_i = 2
    a0: float = 0.0
    mu: float = 0.25
    delta: float = 0.95
    m: int = 15
    xi: float = 0.1
    k: int = 1

    def __post_init__(self) -> None:
        self.c = np.asarray(self.c, dtype=float)
        self.a = np.asarray(self.a, dtype=float)
        if self.c.shape != (self.n,) or self.a.shape != (self.n,):
            raise ValueError("c and a must have length n")

    @property
    def is_symmetric(self) -> bool:
        return bool(np.allclose(self.c, self.c[0]) and np.allclose(self.a, self.a[0]))

    @classmethod
    def baseline(cls, n: int = 2) -> "EconParams":
        """The paper's Section II.E symmetric duopoly baseline (matches the
        paper exactly at n=2; for n>2 it extends symmetrically with the same
        a_i-c_i=1, used for the Section V.A robustness exercise)."""
        return cls(n=n, c=np.ones(n), a=np.full(n, 2.0), a0=0.0, mu=0.25, delta=0.95, m=15, xi=0.1, k=1)


def demand(prices: np.ndarray, a: np.ndarray, a0: float, mu: float) -> np.ndarray:
    """Vectorized logit demand. The last axis of prices/a is the product
    dimension n; any leading batch dimensions are allowed.

    Implemented by treating the outside good as an (n+1)-th "product" and
    doing a numerically stable softmax over all of them, to avoid exp()
    overflow.
    """
    prices = np.asarray(prices, dtype=float)
    a = np.broadcast_to(a, prices.shape)
    exponent_inside = (a - prices) / mu
    outside_shape = exponent_inside.shape[:-1] + (1,)
    exponent_outside = np.full(outside_shape, a0 / mu, dtype=float)
    all_exp = np.concatenate([exponent_inside, exponent_outside], axis=-1)
    shifted = all_exp - np.max(all_exp, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    shares = exps / exps.sum(axis=-1, keepdims=True)
    return shares[..., :-1]


def profits(prices: np.ndarray, c: np.ndarray, a: np.ndarray, a0: float, mu: float) -> np.ndarray:
    q = demand(prices, a, a0, mu)
    return (prices - c) * q


def _nash_focs(prices: np.ndarray, c: np.ndarray, a: np.ndarray, a0: float, mu: float) -> np.ndarray:
    q = demand(prices, a, a0, mu)
    return q - (1.0 / mu) * q * (1.0 - q) * (prices - c)


def _monopoly_focs(prices: np.ndarray, c: np.ndarray, a: np.ndarray, a0: float, mu: float) -> np.ndarray:
    q = demand(prices, a, a0, mu)
    total_profit = np.sum((prices - c) * q)
    return q * (1.0 + (total_profit - (prices - c)) / mu)


def _symmetric_q(p: float, c0: float, a0_val: float, a_out: float, mu: float, n: int) -> float:
    e = np.exp((a0_val - p) / mu)
    e_out = np.exp(a_out / mu)
    return e / (n * e + e_out)


def _solve_symmetric(params: EconParams, monopoly: bool) -> float:
    """Scalar solver for the symmetric case, used as an independent
    cross-check for fsolve (not the main code path)."""
    c0, a0_val = params.c[0], params.a[0]
    n = params.n

    def residual(p: float) -> float:
        q = _symmetric_q(p, c0, a0_val, params.a0, params.mu, n)
        denom = (1.0 - n * q) if monopoly else (1.0 - q)
        return (p - c0) - params.mu / denom

    lo, hi = c0 + 1e-6, c0 + 200.0
    # Scan for a sign-change interval first, then use brentq — avoids
    # manually guessing a bracket.
    grid = np.linspace(lo, hi, 20000)
    vals = np.array([residual(p) for p in grid])
    sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(sign_changes) == 0:
        raise RuntimeError("No sign-change interval found; check whether the parameters are reasonable")
    idx = sign_changes[0]
    return brentq(residual, grid[idx], grid[idx + 1])


def _best_response(p_other: np.ndarray, i: int, params: EconParams) -> float:
    """Player i's static best response given rivals' prices (1-D numerical
    optimization; doesn't rely on the FOC system, so it can't be led astray
    to a spurious stationary point the way solving the simultaneous
    equations can)."""
    def neg_profit(pi: np.ndarray) -> float:
        prices = p_other.copy()
        prices[i] = pi[0]
        return -profits(prices, params.c, params.a, params.a0, params.mu)[i]

    res = minimize(neg_profit, x0=[p_other[i]], bounds=[(params.c[i] + 1e-6, params.c[i] + 50.0)], method="L-BFGS-B")
    return float(res.x[0])


def _tatonnement_nash(params: EconParams, tol: float = 1e-10, max_iter: int = 2000) -> np.ndarray:
    """Best-response dynamics (Gauss-Seidel-style alternating best responses)
    converging to the Nash equilibrium. This is the main code path for
    solving the (possibly asymmetric) logit Bertrand-Nash equilibrium: it is
    far more robust than solving the first-order-condition system directly —
    that FOC system can have multiple stationary points for this problem
    (e.g. the residual can also approach 0 as prices diverge to very large
    values), and least_squares/fsolve are prone to being pulled toward one of
    those non-equilibrium solutions. This was hit in practice during
    development on a cost-asymmetric example (the residual looked tiny, but
    the solution was an obviously unreasonable price vector like [6.9, 1.7];
    switching to best-response iteration gave the correct [1.40, 1.30]
    instead).
    """
    prices = params.c + 1.0
    for _ in range(max_iter):
        new_prices = prices.copy()
        for i in range(params.n):
            new_prices[i] = _best_response(new_prices, i, params)
        if np.max(np.abs(new_prices - prices)) < tol:
            return new_prices
        prices = new_prices
    raise RuntimeError("Best-response iteration did not converge within max_iter")


def solve_nash(params: EconParams) -> np.ndarray:
    """Static Bertrand-Nash price vector p^N. Uses the scalar brentq solver
    in the symmetric case (fastest and most robust); uses best-response
    iteration (tâtonnement) in the asymmetric case — see `_tatonnement_nash`
    for why.
    """
    if params.is_symmetric:
        p_star = _solve_symmetric(params, monopoly=False)
        return np.full(params.n, p_star)
    return _tatonnement_nash(params)


def solve_monopoly(params: EconParams, x0: np.ndarray | None = None) -> np.ndarray:
    """Joint-profit-maximizing price vector p^M. Uses the scalar brentq
    solver in the symmetric case. In the asymmetric case, uses bounded
    numerical optimization to directly maximize joint profit (more robust
    than solving the FOC system — the FOC can also be "satisfied" to within
    numerical precision as prices diverge to infinity, which can mislead
    fsolve; see the tâtonnement docstring above for the same issue in a
    related context).
    """
    if params.is_symmetric:
        p_star = _solve_symmetric(params, monopoly=True)
        return np.full(params.n, p_star)
    if x0 is None:
        x0 = params.c + 1.0

    def neg_joint_profit(prices: np.ndarray) -> float:
        return -float(np.sum(profits(prices, params.c, params.a, params.a0, params.mu)))

    bounds = [(ci + 1e-6, ci + 50.0) for ci in params.c]
    res = minimize(neg_joint_profit, x0, bounds=bounds, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"Monopoly price solver did not converge: {res.message}")
    return res.x


def build_price_grid(params: EconParams, p_nash: np.ndarray, p_monopoly: np.ndarray) -> np.ndarray:
    """Discretization from the paper's Section II.B: each player's feasible
    price set is m equally spaced points on
    [p^N - xi*(p^M-p^N), p^M + xi*(p^M-p^N)].

    Returns an array of shape (n, m); row i is player i's price grid. All
    rows are identical under the symmetric baseline.
    """
    lo = p_nash - params.xi * (p_monopoly - p_nash)
    hi = p_monopoly + params.xi * (p_monopoly - p_nash)
    grids = np.array([np.linspace(lo[i], hi[i], params.m) for i in range(params.n)])
    return grids


def build_profit_matrix(params: EconParams, grids: np.ndarray) -> np.ndarray:
    """Precompute profits for every discrete action combination, shape =
    (m,)*n + (n,).

    This is the key to making the simulation fast: the Q-learning main loop
    does a table lookup instead of recomputing exp(). Under the symmetric
    duopoly baseline this is a small 15x15x2 array.
    """
    n, m = params.n, params.m
    # meshgrid generates every price combination
    mesh = np.meshgrid(*[grids[i] for i in range(n)], indexing="ij")
    price_combos = np.stack(mesh, axis=-1)  # shape (m,)*n + (n,)
    profit_mat = profits(price_combos, params.c, params.a, params.a0, params.mu)
    return profit_mat  # shape (m,)*n + (n,)


def margin(price: float, cost: float) -> float:
    """The "price-cost margin" the paper's text refers to is (p-c)/c (markup
    relative to cost), not the Lerner index (p-c)/p. Verified against the
    baseline parameters: (p^N-c)/c ≈ 0.473 ≈ 47%, (p^M-c)/c ≈ 0.925, about
    twice the former — consistent with the paper's description in Section
    II.E."""
    return (price - cost) / cost
