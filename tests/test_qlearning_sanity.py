"""
纯 Q-learning 实现正确性检验，完全脱离双寡头定价博弈的经济环境——用两个"已知
解析解/可独立数值求解"的小型 MDP 来测，这样如果测试失败，问题一定出在
qlearning.py 本身（更新公式、epsilon-greedy、状态索引），而不会和 environment.py
的经济学建模混在一起，方便定位 bug。

测试 1：单状态老虎机（自循环）。这个特例可以手推出闭式解：
    Q*(s,a) = r_a + delta * r_max / (1-delta)
（含义：选 a 拿一次性奖励 r_a，之后永远按最优动作走，贴现价值是 r_max/(1-delta)）
不依赖 Q-learning 或价值迭代的数值实现，是纯解析的基准。

测试 2：3 状态、3 动作、确定性转移的随机 MDP（固定种子）。这里状态会真正变化，
足以检验 Bellman 更新里 "用 next_state 的 max_a' Q(next_state,a')" 这一步有没
有写对（比如状态索引错位这种 bug，在单状态测试里是测不出来的）。真值用价值
迭代（value iteration）独立算出来，和 Q-learning 训练出的 Q 矩阵对比。

运行：python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

import numpy as np

from src.qlearning import greedy_policy, train_single_agent


class TestBanditClosedForm(unittest.TestCase):
    """单状态、确定性奖励的老虎机，Q* 有解析解。"""

    def setUp(self) -> None:
        self.rewards = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        self.delta = 0.9

        def bandit_step(state, action, rng):
            return self.rewards[action], state  # 自循环：回到唯一的状态

        self.step_fn = bandit_step

    def test_converges_to_closed_form_q_values(self) -> None:
        rng = np.random.default_rng(42)
        Q = train_single_agent(
            step_fn=self.step_fn,
            state_shape=(1,),
            n_actions=5,
            alpha=0.05,
            beta=2e-5,
            delta=self.delta,
            n_periods=300_000,
            rng=rng,
            initial_state=(0,),
        )
        r_max = self.rewards.max()
        q_star = self.rewards + self.delta * r_max / (1.0 - self.delta)
        np.testing.assert_allclose(Q[0], q_star, atol=1e-6)

    def test_greedy_policy_picks_true_optimal_arm(self) -> None:
        rng = np.random.default_rng(1)
        Q = train_single_agent(
            step_fn=self.step_fn,
            state_shape=(1,),
            n_actions=5,
            alpha=0.05,
            beta=2e-5,
            delta=self.delta,
            n_periods=300_000,
            rng=rng,
            initial_state=(0,),
        )
        self.assertEqual(int(greedy_policy(Q)[0]), int(np.argmax(self.rewards)))


class TestMultiStateMDPAgainstValueIteration(unittest.TestCase):
    """3 状态、3 动作、确定性转移的随机 MDP（固定种子生成），真值用价值迭代
    独立算出（不复用 qlearning.py 里的任何函数），验证 Q-learning 收敛的极限
    和真正的 Bellman 最优 Q 函数一致。"""

    def setUp(self) -> None:
        self.n_states, self.n_actions = 3, 3
        self.delta = 0.9
        gen = np.random.default_rng(7)
        self.T = gen.integers(0, self.n_states, size=(self.n_states, self.n_actions))
        self.R = gen.uniform(-1, 1, size=(self.n_states, self.n_actions))
        self.Q_star = self._value_iteration()

        def step_fn(state, action, rng):
            s = state[0]
            reward = self.R[s, action]
            next_state = (int(self.T[s, action]),)
            return reward, next_state

        self.step_fn = step_fn

    def _value_iteration(self) -> np.ndarray:
        V = np.zeros(self.n_states)
        for _ in range(10_000):
            Q_vi = self.R + self.delta * V[self.T]
            V_new = Q_vi.max(axis=1)
            if np.max(np.abs(V_new - V)) < 1e-13:
                V = V_new
                break
            V = V_new
        return self.R + self.delta * V[self.T]

    def test_q_learning_matches_value_iteration(self) -> None:
        rng = np.random.default_rng(123)
        Q = train_single_agent(
            step_fn=self.step_fn,
            state_shape=(self.n_states,),
            n_actions=self.n_actions,
            alpha=0.05,
            beta=1e-5,
            delta=self.delta,
            n_periods=500_000,
            rng=rng,
            initial_state=(0,),
        )
        np.testing.assert_allclose(Q, self.Q_star, atol=1e-3)

    def test_greedy_policy_matches_optimal_policy(self) -> None:
        rng = np.random.default_rng(999)
        Q = train_single_agent(
            step_fn=self.step_fn,
            state_shape=(self.n_states,),
            n_actions=self.n_actions,
            alpha=0.05,
            beta=1e-5,
            delta=self.delta,
            n_periods=500_000,
            rng=rng,
            initial_state=(0,),
        )
        optimal_policy = np.argmax(self.Q_star, axis=1)
        np.testing.assert_array_equal(greedy_policy(Q), optimal_policy)


if __name__ == "__main__":
    unittest.main()
