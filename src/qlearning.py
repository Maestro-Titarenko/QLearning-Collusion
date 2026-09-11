"""
Q-learning 核心原语：Q 矩阵初始化、更新规则、epsilon-greedy 探索。

对应 Calvano et al. (2020) Section I.A（通用 Q-learning）与 Section II.D-E
（应用到重复定价博弈时的具体设定）。方程编号见 CLAUDE.md 第 4-6 节。

状态表示约定：不做手动的混合进制编码，直接用 NumPy 的多维数组 + 元组索引。
对 n 个玩家、k 期记忆、m 个离散价格：
    Q_i 的 shape = (m,)*(n*k) + (m,)
前 n*k 维是状态（按固定顺序排列的过去 k 期、n 个玩家的价格索引），最后一维是
当前玩家自己要选的动作（价格索引）。这样索引 Q_i[state_tuple] 直接给出该状态
下所有动作的 Q 值向量，argmax/max 都是普通 NumPy 操作，不需要额外的编解码层。
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def epsilon_greedy_rate(t: int, beta: float) -> float:
    """式 (7)：epsilon_t = exp(-beta * t)。"""
    return float(np.exp(-beta * t))


def init_q_matrix(profit_matrix: np.ndarray, delta: float, n: int, m: int, k: int) -> np.ndarray:
    """式 (8)：Q_{i,0}(s, a_i) = [ 假设对手均匀随机出价时 i 选 a_i 的期望单期利润 ] / (1-delta)。

    这个初始值不依赖 s（对每个状态都一样），因为在 t=0 时刻还没有任何历史信息
    可以用来区分不同状态。

    参数
    ----
    profit_matrix : shape (m,)*n + (n,)，build_profit_matrix() 的输出。
    delta, n, m, k : 见 EconParams。

    返回
    ----
    shape (n,) + (m,)*(n*k) + (m,) 的数组。
    """
    q0 = np.empty((n, m))
    for i in range(n):
        other_axes = tuple(ax for ax in range(n) if ax != i)
        avg_profit_given_ai = profit_matrix[..., i].mean(axis=other_axes)  # shape (m,)
        q0[i] = avg_profit_given_ai / (1.0 - delta)

    state_shape = (m,) * (n * k)
    Q = np.empty((n,) + state_shape + (m,))
    for i in range(n):
        # 广播：q0[i] 的 shape 是 (m,)，与 Q[i] 的最后一维（动作维）对齐，
        # 自动填满所有状态维。
        Q[i, ...] = q0[i]
    return Q


def greedy_action(q_values: np.ndarray) -> int:
    """在给定状态的 Q 值向量里选最优动作。并列时 np.argmax 默认取第一个（=索引
    最小），而我们的价格网格是严格递增的，所以这正好对应论文的 tie-break 规则
    ——并列时选最低价格。"""
    return int(np.argmax(q_values))


def choose_action(Q_i: np.ndarray, state: Tuple[int, ...], t: int, beta: float, m: int, rng: np.random.Generator) -> int:
    """epsilon-greedy：以 epsilon_t 概率均匀随机探索，否则贪婪选择。"""
    eps = epsilon_greedy_rate(t, beta)
    if rng.random() < eps:
        return int(rng.integers(0, m))
    return greedy_action(Q_i[state])


def update_q(Q_i: np.ndarray, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...], alpha: float, delta: float) -> None:
    """式 (4)：就地更新 Q_i 里被访问到的那一个 (state, action) 格子。"""
    idx = state + (action,)
    best_next = np.max(Q_i[next_state])
    td_target = reward + delta * best_next
    Q_i[idx] = (1.0 - alpha) * Q_i[idx] + alpha * td_target


def greedy_policy(Q_i: np.ndarray) -> np.ndarray:
    """返回每个状态下的贪婪动作，shape = Q_i.shape[:-1]（即状态维那部分）。
    用于收敛判据：比较连续多期这个策略是否变化。"""
    return np.argmax(Q_i, axis=-1)


# ---------------------------------------------------------------------------
# 通用单智能体 Q-learning 训练器：与具体经济环境完全解耦，只依赖一个状态转移
# 函数 step(state, action) -> (reward, next_state)。用途：
#   1) tests/test_qlearning_sanity.py 里在已知解析解的小型 MDP 上验证实现正确性
#   2) 未来任何单智能体基准测试都可以复用
# 重复博弈（多智能体同时学习）的驱动逻辑放在 simulate.py 里，因为那里每期要
# 同时推进两个（或更多）agent 并处理联合状态转移，和这里的单智能体循环结构不同。
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
    """跑一个单智能体、有限期数的 Q-learning 训练循环，返回最终 Q 矩阵。

    step_fn(state, action, rng) -> (reward, next_state) 封装了环境动态，训练
    器本身不关心这是什么环境（可以是无状态老虎机、也可以是多状态 MDP）。
    """
    Q = np.full(state_shape + (n_actions,), q_init_value, dtype=float)
    state = initial_state
    for t in range(n_periods):
        action = choose_action(Q, state, t, beta, n_actions, rng)
        reward, next_state = step_fn(state, action, rng)
        update_q(Q, state, action, reward, next_state, alpha, delta)
        state = next_state
    return Q
