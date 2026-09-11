"""
environment.py 的验收测试。核心思路：用论文原文给出的具体数字（Section II.E
"the price-cost margin is approximately 47 percent... about twice as large
under perfect collusion"）作为独立于我们自己推导的外部基准，而不是只检查代码
内部自洽。

本环境无法访问 PyPI，装不了 pytest，所以用标准库 unittest。
运行：python3 -m unittest discover -s tests -v
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
    """对照论文原文数字的验收测试（Section II.E）。"""

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
        # 已知的第三方复现代码给出的基准 Nash 价格 ~1.4729970（独立交叉验证）
        self.assertAlmostEqual(self.pN[0], 1.4729970, delta=1e-4)

    def test_monopoly_price_matches_known_replication_value(self) -> None:
        self.assertAlmostEqual(self.pM[0], 1.9249689, delta=1e-4)


class TestFOCConsistency(unittest.TestCase):
    """一阶条件在求得的均衡点上应当（近似）为零，无论对称还是非对称。"""

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
        """把对称情形强制走非对称求解路径（tâtonnement），应该和标量闭式解一致——
        这是两条独立代码路径的交叉验证。"""
        params = EconParams.baseline(n=2)
        p_closed_form = solve_nash(params)
        p_tatonnement = _tatonnement_nash(params)
        # L-BFGS-B 内层最优反应求解的默认精度比 brentq 松，两条路径能对齐到
        # 1e-5 量级已经足以说明它们在解同一个均衡，不需要机器精度级别的相等
        np.testing.assert_allclose(p_closed_form, p_tatonnement, atol=1e-4)


class TestAsymmetricSolverSanity(unittest.TestCase):
    """回归测试：开发过程中 least_squares 解 FOC 联立方程组一度收敛到虚假驻点
    （成本不对称例子给出 [6.93, 1.72] 这种明显不合理的价格）。改用最优反应迭代
    后应恒定复现下面这组用独立的两点式 best-response 迭代核实过的数值。"""

    def test_cost_asymmetric_nash_price_is_reasonable(self) -> None:
        params = EconParams(n=2, c=np.array([1.0, 0.7]), a=np.array([2.0, 2.0]), a0=0.0, mu=0.25, delta=0.95)
        pN = solve_nash(params)
        np.testing.assert_allclose(pN, [1.40456, 1.29904], atol=1e-4)
        # 明确排除曾经出现过的虚假解所在的量级
        self.assertLess(pN[0], 3.0)


class TestNumberOfFirmsComparativeStatics(unittest.TestCase):
    """更多企业 -> 竞争更激烈 -> 静态 Nash 价格（加成）应该下降，这是标准寡占
    比较静态结果，用来检查一般化到 n>2 没有引入符号错误。"""

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
        self.assertLess(np.sum(q), 1.0)  # 剩余份额被外部选择占据

    def test_price_grid_bounds_and_length(self) -> None:
        params = EconParams.baseline(n=2)
        pN, pM = solve_nash(params), solve_monopoly(params)
        grids = build_price_grid(params, pN, pM)
        self.assertEqual(grids.shape, (2, 15))
        expected_lo = pN[0] - params.xi * (pM[0] - pN[0])
        expected_hi = pM[0] + params.xi * (pM[0] - pN[0])
        self.assertAlmostEqual(grids[0, 0], expected_lo)
        self.assertAlmostEqual(grids[0, -1], expected_hi)
        self.assertTrue(np.all(np.diff(grids[0]) > 0))  # 严格递增

    def test_profit_matrix_matches_direct_computation(self) -> None:
        params = EconParams.baseline(n=2)
        pN, pM = solve_nash(params), solve_monopoly(params)
        grids = build_price_grid(params, pN, pM)
        profit_mat = build_profit_matrix(params, grids)
        self.assertEqual(profit_mat.shape, (15, 15, 2))
        # 随机抽几个格子核对是否等于直接调用 profits() 的结果
        rng = np.random.default_rng(0)
        for _ in range(10):
            i, j = rng.integers(0, 15, size=2)
            direct = profits(np.array([grids[0, i], grids[1, j]]), params.c, params.a, params.a0, params.mu)
            np.testing.assert_allclose(profit_mat[i, j], direct, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
