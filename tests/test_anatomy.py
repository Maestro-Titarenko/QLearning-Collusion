"""
src/anatomy.py 的验收测试。核心思路和 test_qlearning_sanity.py 一样：构造一个
手算得出解析解的简单场景（这里是"策略永远选同一个动作"的常数价格情形），
不依赖训练过程，只测偏离分析本身的数学逻辑（状态编解码、精确贴现求和、
静态最优反应）。

运行：python3 -m unittest discover -s tests -v
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
    """策略如果对所有状态都选同一个动作组合，价格永远不变（极限环长度=1），
    贴现利润有解析解 profit/(1-delta)，用来验证 discounted_value_from 的
    "瞬态+循环"分解没有算错。"""

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
        constant_action = np.array([7, 7])  # 网格中点附近随便挑一个动作组合
        policy = np.tile(constant_action[:, None], (1, S))  # 不管什么状态都选同一个动作

        first_state = int(np.dot(constant_action, self.strides))
        V = discounted_value_from(policy, self.profit_matrix_flat, self.strides, self.params.delta, first_state, max_steps=S + 1)

        expected_profit = self.profit_matrix_flat[first_state]
        expected_V = expected_profit / (1.0 - self.params.delta)
        np.testing.assert_allclose(V, expected_V, atol=1e-10)

    def test_deviation_from_constant_policy_is_unprofitable(self) -> None:
        """在一个（人为构造的）"永远合谋在垄断价格附近"的常数策略下，偏离到
        静态最优反应应该在当期有利可图，但因为这个简单策略里偏离之后大家还是
        立刻恢复常数策略（没有真正的惩罚机制），偏离在贴现利润上应该是划算
        的——这是用来确认"划算/不划算"的判断逻辑本身没有反过来。
        """
        m, S = self.params.m, self.params.m ** self.params.n
        high_price_action = np.array([13, 13])  # 网格里偏高的价格，比较接近垄断
        policy = np.tile(high_price_action[:, None], (1, S))

        s0 = int(np.dot(high_price_action, self.strides))
        outcome = analyze_deviation(
            policy, self.profit_matrix_full, self.profit_matrix_flat, self.strides, self.params.delta,
            s0, deviating_player=0, n=2, m=m, horizon=10, max_steps=S + 1,
        )
        # 没有惩罚机制的策略下，打一次最优反应赚一次多的钱、之后立刻无痛回到
        # 原来的合谋价格，偏离必然是划算的
        self.assertTrue(outcome.profitable)
        self.assertGreater(outcome.pct_gain_deviator, 0)


class TestStaticBestResponse(unittest.TestCase):
    def test_best_response_improves_on_matching_rival_price(self) -> None:
        params = EconParams.baseline(n=2)
        p_nash, p_monopoly = solve_nash(params), solve_monopoly(params)
        grids = build_price_grid(params, p_nash, p_monopoly)
        profit_matrix_full = build_profit_matrix(params, grids)

        rival_action = 10  # 对手打一个偏高的价格
        best = static_best_response_2p(profit_matrix_full, deviating_player=0, rival_action=rival_action)
        # 最优反应下的利润应该 >= 直接"照抄"对手价格的利润（不是全局最优的
        # 严格证明，只是一个很弱、但能抓出符号错误的合理性检验）
        profit_best = profit_matrix_full[best, rival_action, 0]
        profit_matching = profit_matrix_full[rival_action, rival_action, 0]
        self.assertGreaterEqual(profit_best, profit_matching)


if __name__ == "__main__":
    unittest.main()
