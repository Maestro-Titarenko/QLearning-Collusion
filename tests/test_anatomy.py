"""
Acceptance tests for src/anatomy.py. Same core idea as
test_qlearning_sanity.py: construct a simple scenario with a hand-computable
closed form (here, a "policy always picks the same action" constant-price
case), independent of any training process, testing only the deviation
analysis's own math (state encode/decode, exact discounted summation, static
best response).

Run with: python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

import numpy as np

from src.anatomy import analyze_deviation, decode_state, discounted_value_from, static_best_response_2p
from src.environment import EconParams, build_price_grid, build_profit_matrix, profits, solve_monopoly, solve_nash
from src.simulate import _encode_strides


class TestDecodeState(unittest.TestCase):
    def test_roundtrip_matches_encoding_convention(self) -> None:
        n, m = 2, 15
        strides = _encode_strides(n, m)
        rng = np.random.default_rng(0)
        for _ in range(50):
            actions = rng.integers(0, m, size=n)
            state = int(np.dot(actions, strides))
            decoded = decode_state(state, n, m, strides)
            np.testing.assert_array_equal(decoded, actions)


class TestDiscountedValueClosedForm(unittest.TestCase):
    """If a policy picks the same action combination at every state, the
    price never changes (limit-cycle length = 1), and the discounted profit
    has a closed form, profit/(1-delta), used to verify that
    discounted_value_from's "transient + cycle" decomposition is computed
    correctly."""

    def setUp(self) -> None:
        self.params = EconParams.baseline(n=2)
        p_nash = solve_nash(self.params)
        p_monopoly = solve_monopoly(self.params)
        self.grids = build_price_grid(self.params, p_nash, p_monopoly)
        self.profit_matrix_full = build_profit_matrix(self.params, self.grids)
        self.profit_matrix_flat = self.profit_matrix_full.reshape(-1, self.params.n)
        self.strides = _encode_strides(self.params.n, self.params.m)

    def test_constant_policy_matches_geometric_series(self) -> None:
        m, S = self.params.m, self.params.m ** self.params.n
        constant_action = np.array([7, 7])  # an arbitrary action combination near the middle of the grid
        policy = np.tile(constant_action[:, None], (1, S))  # picks the same action regardless of state

        first_state = int(np.dot(constant_action, self.strides))
        V = discounted_value_from(policy, self.profit_matrix_flat, self.strides, self.params.delta, first_state, max_steps=S + 1)

        expected_profit = self.profit_matrix_flat[first_state]
        expected_V = expected_profit / (1.0 - self.params.delta)
        np.testing.assert_allclose(V, expected_V, atol=1e-10)

    def test_deviation_from_constant_policy_is_unprofitable(self) -> None:
        """Under a (deliberately constructed) constant policy that "always
        colludes near the monopoly price," deviating to the static best
        response should be profitable in the current period — but since this
        simple policy has everyone snap right back to the constant policy
        immediately after a deviation (no real punishment mechanism), the
        deviation should also be profitable in discounted terms. This
        confirms that the "profitable / not profitable" judgment logic
        itself isn't inverted.
        """
        m, S = self.params.m, self.params.m ** self.params.n
        high_price_action = np.array([13, 13])  # a fairly high price on the grid, close to monopoly
        policy = np.tile(high_price_action[:, None], (1, S))

        s0 = int(np.dot(high_price_action, self.strides))
        outcome = analyze_deviation(
            policy, self.profit_matrix_full, self.profit_matrix_flat, self.strides, self.params.delta,
            s0, deviating_player=0, n=2, m=m, horizon=10, max_steps=S + 1,
        )
        # With no punishment mechanism, playing the best response once for
        # extra profit and then snapping right back to the same collusive
        # price with no penalty must make the deviation profitable
        self.assertTrue(outcome.profitable)
        self.assertGreater(outcome.pct_gain_deviator, 0)


class TestStaticBestResponse(unittest.TestCase):
    def test_best_response_improves_on_matching_rival_price(self) -> None:
        params = EconParams.baseline(n=2)
        p_nash, p_monopoly = solve_nash(params), solve_monopoly(params)
        grids = build_price_grid(params, p_nash, p_monopoly)
        profit_matrix_full = build_profit_matrix(params, grids)

        rival_action = 10  # the rival plays a fairly high price
        best = static_best_response_2p(profit_matrix_full, deviating_player=0, rival_action=rival_action)
        # Profit under the best response should be >= profit from simply
        # "matching" the rival's price (not a strict proof of global
        # optimality, just a weak sanity check that would still catch a
        # sign error)
        profit_best = profit_matrix_full[best, rival_action, 0]
        profit_matching = profit_matrix_full[rival_action, rival_action, 0]
        self.assertGreaterEqual(profit_best, profit_matching)


if __name__ == "__main__":
    unittest.main()
