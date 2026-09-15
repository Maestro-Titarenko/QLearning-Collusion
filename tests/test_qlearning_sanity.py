"""
Correctness tests for the pure Q-learning implementation, completely
decoupled from the duopoly pricing-game economic environment — tested
against two small MDPs with "known closed-form or independently
computable" solutions, so that if a test fails, the problem must be in
qlearning.py itself (the update formula, epsilon-greedy, state indexing)
rather than getting tangled up with environment.py's economic modeling,
which makes bugs easier to locate.

Test 1: a single-state bandit (self-loop). This special case has a
hand-derivable closed form:
    Q*(s,a) = r_a + delta * r_max / (1-delta)
(meaning: pick a for a one-time reward r_a, then follow the optimal action
forever after, with discounted value r_max/(1-delta)). This doesn't rely on
any numerical implementation of Q-learning or value iteration — it's a
purely analytic benchmark.

Test 2: a random MDP with 3 states, 3 actions, and deterministic
transitions (fixed seed). Here the state actually changes, which is enough
to check whether the Bellman update's "use max_a' Q(next_state,a') for
next_state" step is implemented correctly (a bug like a misaligned state
index wouldn't show up in the single-state test). The ground truth is
computed independently via value iteration and compared against the Q
matrix Q-learning converges to.

Run with: python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

import numpy as np

from src.qlearning import greedy_policy, train_single_agent


class TestBanditClosedForm(unittest.TestCase):
    """A single-state, deterministic-reward bandit, where Q* has a closed
    form."""

    def setUp(self) -> None:
        self.rewards = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        self.delta = 0.9

        def bandit_step(state, action, rng):
            return self.rewards[action], state  # self-loop: back to the only state

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
    """A random MDP with 3 states, 3 actions, and deterministic transitions
    (generated with a fixed seed); the ground truth is computed
    independently via value iteration (reusing none of qlearning.py's
    functions), verifying that Q-learning's converged limit matches the
    true Bellman-optimal Q-function."""

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
