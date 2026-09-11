# Q-learning 算法合谋复现项目

复现对象：Calvano, Calzolari, Denicolò & Pastorello (2020), "Artificial
Intelligence, Algorithmic Pricing, and Collusion", *American Economic
Review* 110(10): 3267–3297. 原文 PDF 在 `/root/.claude/uploads/.../calvanoetal2020...pdf`。

这份文件是给后续开发（人类或 Claude Code 会话）用的速查表：写代码时应该对照
这里的方程和参数，而不是凭记忆重新推导论文。所有方程编号与论文一致。

## 1. 经济环境（论文 Section II.A）

n 个差异化产品 + 1 个外部选择（outside good）。产品 i 在 t 期的需求（logit）：

    q_i,t = exp((a_i - p_i,t) / mu) / [ sum_j exp((a_j - p_j,t) / mu) + exp(a_0 / mu) ]     (5)

- a_i：产品 i 的垂直质量指数
- a_0：外部选择的（反向）总需求指数
- mu：横向差异化程度，mu -> 0 时退化为完全替代品（Bertrand 悖论）

单期利润：pi_i,t = (p_i,t - c_i) * q_i,t，c_i 为边际成本（固定成本不影响，只要企业留在市场）。

## 2. 动作空间离散化（Section II.B）

对每组参数，先数值求解一次性博弈（static game）的 Bertrand-Nash 价格向量 p^N
和联合利润最大化的垄断价格向量 p^M（对称情形下两者都是标量）。可行价格集 A 是
[p^N - xi*(p^M - p^N),  p^M + xi*(p^M - p^N)] 区间上的 m 个等距点。

基准参数 xi = 0.1，m = 15。

**关键实现优化**：利润 pi_i(a) 只依赖离散动作组合 a = (a_1,...,a_n)（价格索引
组合），不依赖状态。所以可以在 session 开始前把 m^n 种价格组合的利润矩阵一次性
预计算好（对称双寡头下是 15x15 = 225 个格子），运行时每期只是查表，不需要重新
算 exp()。这是把整个模拟做快的核心技巧。

## 3. 状态与记忆（Section II.C）

s_t = {p_{t-1}, ..., p_{t-k}}，k 为记忆长度（基准 k=1，即只记上一期所有玩家的
价格）。状态空间大小 |S| = m^(n*k)，动作空间 |A| = m。

对称双寡头基准：|S| = 15^2 = 225，|A| = 15。每个 agent 的 Q 矩阵是 225 x 15。

## 4. Q-learning（Section I.A 数学定义，Section II 应用到重复博弈）

Q 函数定义（式 3）：

    Q(s,a) = E[pi | s,a] + delta * E[ max_{a'} Q(s',a') | s,a ]

更新规则（式 4），每期只更新被访问到的 (s_t, a_t) 格子：

    Q_{t+1}(s,a) = (1 - alpha) * Q_t(s,a) + alpha * [ pi_t + delta * max_a Q_t(s_{t+1}, a) ]     if s=s_t, a=a_t
    Q_{t+1}(s,a) = Q_t(s,a)                                                                       otherwise

- alpha ∈ [0,1]：学习率，基准网格 [0.025, 0.25]
- delta：贴现因子，基准 0.95

## 5. 探索策略（Section II.D）

epsilon-greedy，探索率随时间指数衰减（式 7）：

    epsilon_t = exp(-beta * t)

beta 越大衰减越快。基准代表性实验点：alpha = 0.15, beta = 4e-6（这是论文正文
里反复用来出图的"代表性实验"参数，不是网格的中点）。

## 6. Q 矩阵初始化（式 8）

t=0 时，Q_{i,0}(s, a_i) 设为"假设对手均匀随机出价"时 i 选 a_i 的贴现期望利润：

    Q_{i,0}(s, a_i) = [ sum_{a_{-i} in A^{n-1}} pi_i(a_i, a_{-i}) ] / [ (1-delta) * |A|^{n-1} ]

初始状态 s_0 在每个 session 开始时随机抽取。

## 7. 收敛判据（Section III.B）

对每个玩家、每个状态，若最优动作 a_i,t(s) = argmax_a Q_{i,t}(a,s) 连续
100,000 期不变，则判定收敛。若到 1,000,000,000 期仍未收敛则停止该 session
（论文用的上限；复现时可以设更低的上限并记录未收敛比例）。

Tie-break 规则：并列时选择最低价格（对称博弈下动作集是纯策略，可能在不可行
的目标价格附近震荡）。

## 8. 基准参数化（Section II.E，全部来自论文原文）

    n = 2（对称双寡头）
    c_i = 1
    a_i - c_i = 1   =>  a_i = 2
    a_0 = 0
    mu = 1/4
    delta = 0.95
    m = 15
    xi = 0.1
    k = 1（一期记忆）

验收基准（论文原文数字，写单元测试时用这些数字做 sanity check）：
- 静态 Bertrand-Nash 均衡下加成（margin）≈ 47%
- 完全合谋（垄断）下加成 ≈ 上述的两倍，约 94%

"代表性实验"（论文 Section IV 起大量图表用的那组参数）：
    alpha = 0.15, beta = 4e-6   （其余同基准）

## 9. 关键产出指标

利润增益（式 9）：

    Delta = (pi_bar - pi^N) / (pi^M - pi^N)

pi_bar 是收敛后的平均单企业利润，pi^N 是静态 Bertrand-Nash 利润，pi^M 是垄断
（联合利润最大化）利润。Delta=0 对应完全竞争，Delta=1 对应完全合谋。

论文报告的关键数字（用来核对复现结果是否在合理范围）：
- 全网格（100x100 的 alpha,beta）上 Delta 落在 70%-90% 之间（Figure 1）
- Table 1 "All" 列：平均 Delta = 0.849，Nash 均衡频率 = 0.505，on-path 平均
  Q-loss = 0.002
- n=3 时 Delta ≈ 0.75（调整 beta 后），n=4 时 Delta ≈ 0.56（基准 beta 网格下）
- 不对称成本（Table 4，c_1=1 固定，c_2 从 1 降到 0.25）：Delta 从 0.849 缓慢降
  到 0.713

## 10. 项目结构与开发约定

    src/environment.py   logit 需求、利润矩阵、Nash/垄断价格数值求解
    src/qlearning.py     Q 矩阵初始化、更新规则、epsilon-greedy、单 session 跑法
    src/simulate.py      多 session 驱动、收敛检测、汇总统计
    configs/*.yaml        参数集（对应论文不同实验设置）
    experiments/*.py      跑具体实验、输出到 results/
    tests/                unittest 测试（本环境无法访问 PyPI，不能装 pytest，
                           一律用标准库 unittest）
    analysis/              用 matplotlib 复现论文图表

开发顺序：先验证 environment.py 的数值解对得上第 9 节的基准数字，再验证
qlearning.py 在一个已知解析解的小 MDP 上收敛正确，然后才跑双寡头对局。跑大规
模网格实验前，先在代表性参数点上用较少 session 数（几十到几百）验证统计量落
在合理范围，再决定要不要扩大规模。

**环境限制**：这个 Claude 会话所在的沙箱无法访问 PyPI（网络策略挋掉了
pypi.org/files.pythonhosted.org 的请求，返回 403），所以 numba 装不上。热循环
先用纯 NumPy 写，性能不够时可以考虑：(a) 把多个 session 沿数组维度做 lockstep
向量化（因为每个 session 结构完全一致，可以把 session 作为 batch 维度整体用
NumPy 数组操作推进，而不是 Python 循环跑 1000 个独立 session），或 (b) 在有网
络访问的环境里另装 numba/joblib 做真正的 JIT + 多进程并行。

## 11. Phase 2 进度记录（2026-09-11）

实测下来，纯 NumPy（无 JIT）实现在这个沙箱里跑到了约 10-11 万期/秒（用了
`src/simulate.py` 里"每期只查表 + 只检查被更新的那个状态的贪婪动作有没有变"
这套优化）。沙箱只有 2 个 CPU 核心，用 `joblib.Parallel(n_jobs=2)` 并行跑
session，单 session 平均约 9-10 秒（含约 170-200 万期才收敛，比论文正文提到
的"网格中点 alpha=0.125, beta=1e-5 平均 85 万期收敛"要慢，因为代表性实验点的
beta=4e-6 比网格中点更小，探索衰减更慢）。

`experiments/run_representative.py` 会把结果按 session 追加写到
`results/representative_experiment.jsonl`（断点续跑：已经跑过的 seed 自动跳
过），可以分批用不同的 `--start --n_sessions` 调用来跑更多 session。

跑了 140 个 session 的结果（对照论文 Table 1"All"列）：

| 指标 | 复现结果 | 论文 |
|---|---|---|
| 平均 Δ | 0.861 | 0.849 |
| Δ 标准差 | 0.102 | 0.112 |
| 收敛比例 | 100% | 接近100% |
| 极限环长度=1（常数价格）占比 | 66.4% | 64.3% |
| 极限环长度=2 占比 | 21.4% | 23.8% |
| 极限环长度≥3 占比 | 12.1% | 11.9% |

注意论文 Table 1 的"All"列是整个 100×100 网格的汇总，我们这里只是单一参数点
（代表性实验点），两者能对得这么近某种程度上比较幸运，不代表网格上每一点都
会这么吻合——后续如果要做 Figure 1/2 那种网格热力图，还是得真的把网格扫一遍
才能验证清楚。

`src/simulate.py` 的 `run_session()` 支持 `record_every` 参数记录训练过程中的
窗口平均利润轨迹，用来画类似论文 Figure 10 的学习曲线（见
`analysis/plot_representative.py`）。

## 12. Phase 3 进度记录（2026-09-11）：合谋解剖 / 偏离-惩罚分析

`src/anatomy.py` 实现了论文第四节的核心操作：给定收敛后的极限策略（固定、
不再探索），在极限环的某个状态 s0 外生强制一方玩家在 τ=1 打静态最优反应
（对对手当期价格的最优反应，不是随便降价），τ=2 起双方恢复正常策略。因为
固定策略后系统是确定性有限自动机，价格路径最终一定进入某个循环（鸽笼原
理），所以贴现利润是精确算的（瞬态段 + 循环段解析求和），不是截断近似。

`experiments/run_anatomy.py` 训练 30 个 session（seed 0-29，和 Phase 2 用同一
套 alpha/beta，但这次保留了 `SessionResult.policy` 以复用），对每个 session
收敛后的极限环上每个状态、两种偏离方身份都做一次偏离分析，session 内先对
"循环起点 x 偏离方身份"取平均（对应论文脚注："算作这个 session 的一个观
测"），再跨 session 平均。

结果（对照论文 Table 3 / 正文）：

| 指标 | 复现结果 | 论文 |
|---|---|---|
| 偏离的平均贴现利润变化 | -2.82% | 约 -3% 到 -4% |
| 偏离不划算的比例 | 98.9% | >95% |

`analysis/plot_anatomy.py` 画出的脉冲响应图（`results/anatomy_impulse_response.png`）
和论文 Figure 4 的形状高度吻合：τ=1 价格骤降，τ=2 触底（还略低于 τ=1 的水
平，即论文说的"overshooting"），随后逐步回升，约 τ=9-10 期回到长期价格附
近——这正是论文强调的"有限期价格战 + 逐步回归"的 stick-and-carrot 模式，
而不是永久惩罚的 grim-trigger。

下一步（Phase 4）：稳健性子集——不对称成本（Table 4）、n=3/4 玩家。
`environment.py` 的非对称 Nash/monopoly 求解器已经在 Phase 1 测试过，
`simulate.py`/`anatomy.py` 目前假设 n=2（`static_best_response_2p` 硬编码了
双寡头），n=3/4 需要先把偏离分析泛化到任意 n（Phase 2 的训练部分本身是支持
任意 n 的，只是没有专门跑过）。
