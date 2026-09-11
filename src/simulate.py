"""
双寡头重复定价博弈的 session 驱动器。对应论文 Section II.E-III.B。

性能说明：这个沙箱装不上 numba，所以这里手写了一版针对"无 JIT"场景优化过的
纯 NumPy/Python 实现：
  1. 状态用展平后的整数索引（而不是元组），Q 矩阵 reshape 成 (n, S, m)。
  2. 利润矩阵预先展平成 (S, n)：因为 k=1 记忆下，"下一期状态"的编码方式和
     "这一期动作组合"的编码方式是同一套映射，所以 reward 直接就是
     profit_matrix_flat[next_state]，不需要额外的动作编码步骤。
  3. 每个 chunk（比如 20 万期）批量预生成随机数（探索与否的判定、随机探索时
     选哪个动作），而不是每期都调用一次随机数生成器。
  4. 收敛判据做了等价的效率优化：论文的判据是"每个玩家在每个状态下的最优动作
     连续 N 期不变"。因为 Q-learning 每期只会更新被访问到的那一个 (state,action)
     格子，其余格子的 Q 值、进而其余状态的贪婪动作都不可能变化。所以只需要
     在每期更新后检查"这一期被更新的那个状态，它的贪婪动作有没有变"，如果
     连续 N 期都没有任何一次这样的变化，就等价于整个策略连续 N 期不变。这比
     每期重新算一遍完整的 argmax(Q, axis=-1) 快得多。
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
    avg_profit: np.ndarray  # shape (n,)，长期（收敛后）平均单期利润
    delta: float  # 论文式(9)，用所有玩家的平均利润算
    trace_periods: List[int] = field(default_factory=list)
    trace_avg_profit: List[float] = field(default_factory=list)


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
    """和 build_profit_matrix() 里 meshgrid(indexing='ij') 的展平顺序对齐：
    第 0 个玩家的动作是最高位。"""
    return m ** np.arange(n - 1, -1, -1)


def _extract_steady_state(policy: np.ndarray, profit_matrix_flat: np.ndarray, s_start: int, strides: np.ndarray, max_steps: int) -> tuple[List[int], np.ndarray]:
    """从给定状态出发，按固定（贪婪）策略确定性地往前推演，直到状态重复出现，
    从而精确地找出极限环（可能是长度 1 的"常数价格"，也可能是更长的价格循环）。
    有限状态空间下，max_steps = S+1 就足够保证一定会出现重复（鸽笼原理）。
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
    # 理论上不会走到这里（有限状态空间一定会在 S+1 步内出现重复）
    raise RuntimeError("未能在 max_steps 内找到极限环，检查状态空间大小设置是否够大")


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
    """record_every 不为 None 时，额外记录训练过程中的窗口平均利润轨迹（写在
    返回的 SessionResult.trace 里），用来画类似论文 Figure 10 的学习曲线。这
    个记录本身会有一点开销，所以默认关闭，只在跑单个用于画图的示例 session
    时打开。"""
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
    """跑 n_sessions 个独立 session，返回汇总结果。对应论文 Section II 的
    "每组参数一个 experiment，每个 experiment 1000 个 session" 设计（这里
    session 数可配置，方便先用少量 session 做验证）。"""
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
