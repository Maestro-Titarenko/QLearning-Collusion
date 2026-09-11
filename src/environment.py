"""
经济环境：logit 需求、利润、静态 Bertrand-Nash / 垄断价格求解、动作空间离散化。

对应 Calvano et al. (2020) Section II.A-II.E。方程编号见 CLAUDE.md 第 1-2 节。

FOC 推导（供代码审阅时核对，推导过程见项目笔记，此处只给结论）：

    q_i(p) = exp((a_i - p_i)/mu) / [ sum_j exp((a_j - p_j)/mu) + exp(a0/mu) ]

    dq_i/dp_i = -(1/mu) * q_i * (1 - q_i)
    dq_j/dp_i = (1/mu) * q_i * q_j        (j != i)

    Bertrand-Nash（每个企业只控制自己的价格）一阶条件：
        FOC_i = q_i - (1/mu) * q_i * (1-q_i) * (p_i - c_i) = 0

    垄断 / 联合利润最大化（假想的多产品垄断者同时控制所有价格）一阶条件，化简后：
        FOC_i = q_i * (1 + (S - (p_i - c_i)) / mu) = 0,   S = sum_j (p_j - c_j) * q_j

    对称情形（n 个产品完全对称）退化为标量方程：
        Nash:     p - c = mu / (1 - q(p))
        Monopoly: p - c = mu / (1 - n * q(p))
    这两个标量方程用来做独立于 fsolve 的交叉验证（见 tests/test_environment.py）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize


@dataclass
class EconParams:
    """一组经济环境参数。默认值为论文 baseline（Section II.E）。"""

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
            raise ValueError("c, a 的长度必须等于 n")

    @property
    def is_symmetric(self) -> bool:
        return bool(np.allclose(self.c, self.c[0]) and np.allclose(self.a, self.a[0]))

    @classmethod
    def baseline(cls, n: int = 2) -> "EconParams":
        """论文 Section II.E 的对称双寡头基准（n=2 时与论文完全一致；
        n>2 时按同样的 a_i-c_i=1 对称扩展，用于 Section V.A 的稳健性练习）。"""
        return cls(n=n, c=np.ones(n), a=np.full(n, 2.0), a0=0.0, mu=0.25, delta=0.95, m=15, xi=0.1, k=1)


def demand(prices: np.ndarray, a: np.ndarray, a0: float, mu: float) -> np.ndarray:
    """向量化 logit 需求。prices, a 的最后一维是产品维度 n，可以带任意 batch 维。

    用把外部选择当作第 n+1 个"产品"、整体做数值稳定 softmax 的方式实现，
    避免 exp() 溢出。
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
    """对称情形下的标量求解，用作 fsolve 的独立交叉验证（不是主路径）。"""
    c0, a0_val = params.c[0], params.a[0]
    n = params.n

    def residual(p: float) -> float:
        q = _symmetric_q(p, c0, a0_val, params.a0, params.mu, n)
        denom = (1.0 - n * q) if monopoly else (1.0 - q)
        return (p - c0) - params.mu / denom

    lo, hi = c0 + 1e-6, c0 + 200.0
    # 扫描找一个变号区间再用 brentq，避免手动猜括号
    grid = np.linspace(lo, hi, 20000)
    vals = np.array([residual(p) for p in grid])
    sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(sign_changes) == 0:
        raise RuntimeError("未找到变号区间，检查参数是否合理")
    idx = sign_changes[0]
    return brentq(residual, grid[idx], grid[idx + 1])


def _best_response(p_other: np.ndarray, i: int, params: EconParams) -> float:
    """给定对手价格，玩家 i 的静态最优反应（一维数值优化，不依赖 FOC 方程组，
    不会像联立方程求解那样被引到虚假驻点）。"""
    def neg_profit(pi: np.ndarray) -> float:
        prices = p_other.copy()
        prices[i] = pi[0]
        return -profits(prices, params.c, params.a, params.a0, params.mu)[i]

    res = minimize(neg_profit, x0=[p_other[i]], bounds=[(params.c[i] + 1e-6, params.c[i] + 50.0)], method="L-BFGS-B")
    return float(res.x[0])


def _tatonnement_nash(params: EconParams, tol: float = 1e-10, max_iter: int = 2000) -> np.ndarray:
    """最优反应动态（Gauss-Seidel 式轮流最优反应）收敛到 Nash 均衡。这是求解
    一般化（含非对称）logit Bertrand-Nash 的主路径：比直接解一阶条件联立方程组
    稳健得多——FOC 方程组在这个问题里可能有多个驻点（例如价格发散到很大时残差
    也趋近于 0），least_squares/fsolve 容易被带到那些不是真正均衡的解上，这一点
    在开发时用一个成本不对称的例子实测踩到过坑（残差看似很小，但价格是
    [6.9, 1.7] 这种明显不合理的解；改用最优反应迭代后核对为 [1.40, 1.30]）。
    """
    prices = params.c + 1.0
    for _ in range(max_iter):
        new_prices = prices.copy()
        for i in range(params.n):
            new_prices[i] = _best_response(new_prices, i, params)
        if np.max(np.abs(new_prices - prices)) < tol:
            return new_prices
        prices = new_prices
    raise RuntimeError("最优反应迭代未在 max_iter 内收敛")


def solve_nash(params: EconParams) -> np.ndarray:
    """静态 Bertrand-Nash 价格向量 p^N。对称情形用标量 brentq（最快最稳健）；
    非对称情形用最优反应迭代（tâtonnement），见 `_tatonnement_nash` 的说明。
    """
    if params.is_symmetric:
        p_star = _solve_symmetric(params, monopoly=False)
        return np.full(params.n, p_star)
    return _tatonnement_nash(params)


def solve_monopoly(params: EconParams, x0: np.ndarray | None = None) -> np.ndarray:
    """联合利润最大化价格向量 p^M。对称情形用标量 brentq。非对称情形改用有界
    数值优化直接最大化联合利润（比解 FOC 方程组更稳健——FOC 在价格趋于无穷时
    也可能"满足"到数值精度内，fsolve 容易被带偏，见开发时的踩坑记录）。
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
        raise RuntimeError(f"垄断价格求解未收敛：{res.message}")
    return res.x


def build_price_grid(params: EconParams, p_nash: np.ndarray, p_monopoly: np.ndarray) -> np.ndarray:
    """论文 Section II.B 的离散化：每个玩家的可行价格集是
    [p^N - xi*(p^M-p^N), p^M + xi*(p^M-p^N)] 上的 m 个等距点。

    返回 shape (n, m) 的数组，第 i 行是玩家 i 的价格网格。对称基准下所有行相同。
    """
    lo = p_nash - params.xi * (p_monopoly - p_nash)
    hi = p_monopoly + params.xi * (p_monopoly - p_nash)
    grids = np.array([np.linspace(lo[i], hi[i], params.m) for i in range(params.n)])
    return grids


def build_profit_matrix(params: EconParams, grids: np.ndarray) -> np.ndarray:
    """预计算所有离散动作组合下的利润，shape = (m,)*n + (n,)。

    这是让模拟快起来的关键：Q-learning 主循环里查表而不是重新算 exp()。
    对称双寡头基准下这是一个 15x15x2 的小数组。
    """
    n, m = params.n, params.m
    # meshgrid 生成所有价格组合
    mesh = np.meshgrid(*[grids[i] for i in range(n)], indexing="ij")
    price_combos = np.stack(mesh, axis=-1)  # shape (m,)*n + (n,)
    profit_mat = profits(price_combos, params.c, params.a, params.a0, params.mu)
    return profit_mat  # shape (m,)*n + (n,)


def margin(price: float, cost: float) -> float:
    """论文正文说的"price-cost margin"是 (p-c)/c（相对成本的加成），不是
    Lerner index (p-c)/p。用基准参数验证过：(p^N-c)/c ≈ 0.473 ≈ 47%，
    (p^M-c)/c ≈ 0.925，约为前者的两倍，与论文 Section II.E 的描述一致。"""
    return (price - cost) / cost
