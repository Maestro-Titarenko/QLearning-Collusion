"""
论文 Section IV"合谋解剖"（Anatomy of Collusion）：偏离-惩罚分析。

核心操作（对照论文原文，Figure 4/5、Table 2/3）：从收敛后的极限环上的某个
状态 s0 出发，外生地强制一方玩家在 τ=1 期偏离到"静态最优反应"（即在对手
仍按 s0 隐含的价格出价的前提下，偏离方选出让自己当期利润最大化的价格），
从 τ=2 期起双方都恢复按各自学到的（固定、不再探索的）策略行动，观察价格
如何演化、偏离在贴现利润上是否划算。

这里的关键简化和 simulate.py 一致：策略固定后，整个系统是一个确定性有限自
动机（状态数 S 有限），所以任何一条从某点出发、按固定策略推演的价格路径最
终一定会进入某个循环（鸽笼原理）。这让我们可以精确地（不是靠截断来近似）
算出无限期贴现利润，而不需要数值截断误差。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def decode_state(state: int, n: int, m: int, strides: np.ndarray) -> np.ndarray:
    """把展平的整数状态解码回每个玩家的价格索引（strides 的定义和
    simulate._encode_strides 一致：玩家 0 是最高位）。"""
    actions = np.empty(n, dtype=int)
    remaining = state
    for idx in range(n):
        actions[idx] = remaining // strides[idx]
        remaining = remaining % strides[idx]
    return actions


def static_best_response_2p(profit_matrix_full: np.ndarray, deviating_player: int, rival_action: int) -> int:
    """双寡头（n=2）情形下，给定对手的价格索引，偏离方的静态最优反应价格索引。
    profit_matrix_full 是 build_profit_matrix() 的原始（未展平）输出，
    shape (m, m, 2)。"""
    if deviating_player == 0:
        profits_given_a0 = profit_matrix_full[:, rival_action, 0]
    elif deviating_player == 1:
        profits_given_a0 = profit_matrix_full[rival_action, :, 1]
    else:
        raise ValueError("这个函数目前只支持 n=2 的双寡头")
    return int(np.argmax(profits_given_a0))


def _decompose_trajectory(policy: np.ndarray, strides: np.ndarray, first_state: int, max_steps: int) -> Tuple[List[int], List[int]]:
    """从 first_state 开始，按固定策略确定性推演，拆成"进入循环前的瞬态部分"
    和"循环部分"。返回 (transient_states, cycle_states)。"""
    visited = {}
    seq: List[int] = []
    state = first_state
    for step in range(max_steps):
        if state in visited:
            cycle_start = visited[state]
            return seq[:cycle_start], seq[cycle_start:]
        visited[state] = step
        seq.append(state)
        state = int(np.dot(policy[:, state], strides))
    raise RuntimeError("未能在 max_steps 内找到极限环")


def discounted_value_from(
    policy: np.ndarray,
    profit_matrix_flat: np.ndarray,
    strides: np.ndarray,
    delta: float,
    first_state: int,
    max_steps: int,
) -> np.ndarray:
    """精确计算：从 first_state（即"第 1 期实际发生的价格状态"）开始，按固定
    策略永远玩下去，每个玩家的无限期贴现利润 sum_{t=0}^inf delta^t * pi(s_t)。
    因为轨迹最终周期性，拆成"瞬态 + 循环"两部分算精确解析和，不做截断近似：
        V = sum_{瞬态} delta^t * r_t  +  delta^{瞬态长度} * (循环内 delta 加权和) / (1 - delta^L)
    """
    transient, cycle = _decompose_trajectory(policy, strides, first_state, max_steps)
    n = profit_matrix_flat.shape[1]
    V = np.zeros(n)
    for t, s in enumerate(transient):
        V += (delta**t) * profit_matrix_flat[s]
    L = len(cycle)
    cycle_val = np.zeros(n)
    for j, s in enumerate(cycle):
        cycle_val += (delta**j) * profit_matrix_flat[s]
    V += (delta ** len(transient)) * cycle_val / (1.0 - delta**L)
    return V


@dataclass
class DeviationOutcome:
    s0: int
    deviating_player: int
    price_path: np.ndarray  # shape (horizon+1, n)，索引 0 是 τ=0（偏离前）
    v_baseline: np.ndarray  # shape (n,)，不偏离时从 τ=1 起的贴现利润
    v_deviation: np.ndarray  # shape (n,)，偏离后从 τ=1 起的贴现利润
    pct_gain_deviator: float  # 偏离方的贴现利润变化百分比（论文 Table 3 Panel A 的量）
    profitable: bool


def analyze_deviation(
    policy: np.ndarray,
    profit_matrix_full: np.ndarray,
    profit_matrix_flat: np.ndarray,
    strides: np.ndarray,
    delta: float,
    s0: int,
    deviating_player: int,
    n: int,
    m: int,
    horizon: int,
    max_steps: int,
) -> DeviationOutcome:
    """对一个具体的 (起始状态 s0, 偏离方) 组合，算出价格脉冲响应路径和偏离
    是否划算。目前 static_best_response_2p 只支持 n=2，所以这个函数也限定
    n=2（和论文 Section IV 的基准实验设定一致）。"""
    if n != 2:
        raise NotImplementedError("目前只实现了双寡头 (n=2) 的偏离分析")

    rival = 1 - deviating_player
    s0_actions = decode_state(s0, n, m, strides)

    # τ=1：偏离方打静态最优反应，对手仍按自己的正常策略出价
    dev_action = static_best_response_2p(profit_matrix_full, deviating_player, s0_actions[rival])
    actions_tau1 = policy[:, s0].copy()
    actions_tau1[deviating_player] = dev_action
    s1_deviation = int(np.dot(actions_tau1, strides))

    # 不偏离的基准：τ=1 起双方都按正常策略走
    s1_baseline = int(np.dot(policy[:, s0], strides))

    v_baseline = discounted_value_from(policy, profit_matrix_flat, strides, delta, s1_baseline, max_steps)
    v_deviation = discounted_value_from(policy, profit_matrix_flat, strides, delta, s1_deviation, max_steps)

    # 价格路径（用于画脉冲响应图）：τ=0 是偏离前的状态，τ=1 是刚才算出的
    # actions_tau1，τ=2 起按正常策略继续推进
    price_path = [s0_actions, actions_tau1]
    state = s1_deviation
    for _ in range(horizon - 1):
        actions = policy[:, state]
        price_path.append(actions.copy())
        state = int(np.dot(actions, strides))
    price_path = np.array(price_path)

    pct_gain = (v_deviation[deviating_player] - v_baseline[deviating_player]) / v_baseline[deviating_player]

    return DeviationOutcome(
        s0=s0,
        deviating_player=deviating_player,
        price_path=price_path,
        v_baseline=v_baseline,
        v_deviation=v_deviation,
        pct_gain_deviator=float(pct_gain),
        profitable=bool(pct_gain > 0),
    )
