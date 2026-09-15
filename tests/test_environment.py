"""
Acceptance tests for environment.py. Core idea: use the concrete numbers
given in the paper itself (Section II.E: "the price-cost margin is
approximately 47 percent... about twice as large under perfect collusion")
as an external benchmark independent of our own derivation, rather than only
checking that the code is internally self-consistent.

This environment has no PyPI access and can't install pytest, so everything
uses the standard-library unittest.
Run with: python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

import numpy as np

from src.environment import (
    EconParams,
    _nash_focs,
    _tatonnement_nash,
    build_price_grid,
    build_profit_matrix,
    demand,
    margin,
    profits,
    solve_monopoly,
    solve_nash,
)


class TestBaselineCalibration(unittest.TestCase):
    """Acceptance tests against the paper's own numbers (Section II.E)."""

    def setUp(self) -> None:
        self.params = EconParams.baseline(n=2)
        self.pN = solve_nash(self.params)
        self.pM = solve_monopoly(self.params)

    def test_nash_margin_is_about_47_percent(self) -> None:
        m = margin(self.pN[0], self.params.c[0])
        self.assertAlmostEqual(m, 0.47, delta=0.01)

    def test_monopoly_margin_is_about_double_nash(self) -> None:
        mN = margin(self.pN[0], self.params.c[0])
        mM = margin(self.pM[0], self.params.c[0])
        self.assertAlmostEqual(mM / mN, 2.0, delta=0.05)

    def test_prices_are_symmetric_across_firms(self) -> None:
        self.assertAlmostEqual(self.pN[0], self.pN[1])
        self.assertAlmostEqual(self.pM[0], self.pM[1])

    def test_monopoly_price_exceeds_nash_price(self) -> None:
        self.assertGreater(self.pM[0], self.pN[0])

    def test_joint_profit_maximized_at_monopoly_not_nash(self) -> None:
        pi_nash = np.sum(profits(self.pN, self.params.c, self.params.a, self.params.a0, self.params.mu))
        pi_monopoly = np.sum(profits(self.pM, self.params.c, self.params.a, self.params.a0, self.params.mu))
        self.assertGreater(pi_monopoly, pi_nash)

    def test_nash_price_matches_known_replication_value(self) -> None:
        # A known third-party replication gives a baseline Nash price of
        # ~1.4729970 (independent cross-check)
        self.assertAlmostEqual(self.pN[0], 1.4729970, delta=1e-4)

    def test_monopoly_price_matches_known_replication_value(self) -> None:
        self.assertAlmostEqual(self.pM[0], 1.9249689, delta=1e-4)


class TestFOCConsistency(unittest.TestCase):
    """The first-order conditions should be (approximately) zero at the
    computed equilibrium, whether symmetric or asymmetric."""

    def test_foc_zero_at_symmetric_nash(self) -> None:
        params = EconParams.baseline(n=2)
        pN = solve_nash(params)
        residual = _nash_focs(pN, params.c, params.a, params.a0, params.mu)
        np.testing.assert_allclose(residual, 0.0, atol=1e-6)

    def test_foc_zero_at_asymmetric_nash(self) -> None:
        params = EconParams(n=2, c=np.array([1.0, 0.7]), a=np.array([2.0, 2.0]), a0=0.0, mu=0.25, delta=0.95)
        pN = solve_nash(params)
        residual = _nash_focs(pN, params.c, params.a, params.a0, params.mu)
        np.testing.assert_allclose(residual, 0.0, atol=1e-5)

    def test_tatonnement_matches_closed_form_for_symmetric_case(self) -> None:
        """Forcing the symmetric case through the asymmetric solve path
        (tâtonnement) should agree with the scalar closed-form solution —
        this is a cross-check between two independent code paths."""
        params = EconParams.baseline(n=2)
        p_closed_form = solve_nash(params)
        p_tatonnement = _tatonnement_nash(params)
        # The inner best-response solve's default L-BFGS-B tolerance is
        # looser than brentq's; the two paths agreeing to about 1e-5 is
        # already enough to show they're solving the same equilibrium —
        # machine-precision equality isn't needed here.
        np.testing.assert_allclose(p_closed_form, p_tatonnement, atol=1e-4)


class TestAsymmetricSolverSanity(unittest.TestCase):
    """Regression test: during development, solving the FOC system with
    least_squares once converged to a spurious stationary point (an
    obviously unreasonable price like [6.93, 1.72] for the cost-asymmetric
    example). After switching to best-response iteration, it should
    consistently reproduce the values below, which were independently
    verified with a separate two-point best-response iteration."""

    def test_cost_asymmetric_nash_price_is_reasonable(self) -> None:
        params = EconParams(n=2, c=np.array([1.0, 0.7]), a=np.array([2.0, 2.0]), a0=0.0, mu=0.25, delta=0.95)
        pN = solve_nash(params)
        np.testing.assert_allclose(pN, [1.40456, 1.29904], atol=1e-4)
        # Explicitly rule out the magnitude of the spurious solution seen before
        self.assertLess(pN[0], 3.0)


class TestNumberOfFirmsComparativeStatics(unittest.TestCase):
    """More firms -> tougher competition -> the static Nash price (markup)
    should fall — a standard oligopoly comparative-statics result, used to
    check that generalizing to n>2 didn't introduce a sign error."""

    def test_nash_margin_decreasing_in_n(self) -> None:
        margins = []
        for n in (2, 3, 4):
            params = EconParams.baseline(n=n)
            pN = solve_nash(params)
            margins.append(margin(pN[0], params.c[0]))
        self.assertTrue(margins[0] > margins[1] > margins[2])


class TestDemandAndProfitMatrix(unittest.TestCase):
    def test_demand_is_positive_and_sums_below_one(self) -> None:
        params = EconParams.baseline(n=2)
        prices = np.array([1.5, 1.5])
        q = demand(prices, params.a, params.a0, params.mu)
        self.assertTrue(np.all(q > 0))
        self.assertLess(np.sum(q), 1.0)  # the remaining share goes to the outside good

    def test_price_grid_bounds_and_length(self) -> None:
        params = EconParams.baseline(n=2)
        pN, pM = solve_nash(params), solve_monopoly(params)
        grids = build_price_grid(params, pN, pM)
        self.assertEqual(grids.shape, (2, 15))
        expected_lo = pN[0] - params.xi * (pM[0] - pN[0])
        expected_hi = pM[0] + params.xi * (pM[0] - pN[0])
        self.assertAlmostEqual(grids[0, 0], expected_lo)
        self.assertAlmostEqual(grids[0, -1], expected_hi)
        self.assertTrue(np.all(np.diff(grids[0]) > 0))  # strictly increasing

    def test_profit_matrix_matches_direct_computation(self) -> None:
        params = EconParams.baseline(n=2)
        pN, pM = solve_nash(params), solve_monopoly(params)
        grids = build_price_grid(params, pN, pM)
        profit_mat = build_profit_matrix(params, grids)
        self.assertEqual(profit_mat.shape, (15, 15, 2))
        # spot-check a few random cells against directly calling profits()
        rng = np.random.default_rng(0)
        for _ in range(10):
            i, j = rng.integers(0, 15, size=2)
            direct = profits(np.array([grids[0, i], grids[1, j]]), params.c, params.a, params.a0, params.mu)
            np.testing.assert_allclose(profit_mat[i, j], direct, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
