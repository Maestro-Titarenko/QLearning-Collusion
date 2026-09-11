"""
复现论文 Figure 1：(alpha, beta) 网格上每个格点的平均利润增益 Delta 热力图。
读取 experiments/run_grid.py 产出的 results/grid/alpha_*.jsonl，按
(alpha_idx, beta_idx) 对 session 取平均，画成热力图。

Delta 是连续量（论文 Section 9 式 9），用单色序列色阶（浅->深蓝，取自 dataviz
技能参考色板的 blue 序列坡），不是分类色，所以不能用默认的 tab10/viridis 之
类的彩虹色。
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_DIR = os.path.join(BASE, "results", "grid")

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_LINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# dataviz 技能参考色板：blue 序列坡 100(浅) -> 700(深)，浅色代表"接近0"。
BLUE_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_sequential",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

PAPER_TABLE1_ALL_MEAN = 0.849  # 论文 Table 1 "All" 列，全网格平均


def load_grid(
    n_alpha: int, n_beta: int, alpha_min: float, alpha_max: float, beta_min: float, beta_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (alphas, betas, delta_mean, n_sessions_per_cell)。

    alphas/betas 直接按 experiments/run_grid.py 里同样的公式
    （linspace/geomspace）算，不从数据文件回填——如果某个格点所有 session 都
    没收敛，数据文件里就没有可用的 alpha/beta 值，回填会漏掉这个格点的坐标。
    """
    alphas = np.linspace(alpha_min, alpha_max, n_alpha)
    betas = np.geomspace(beta_min, beta_max, n_beta)

    delta_mean = np.full((n_alpha, n_beta), np.nan)
    delta_count = np.zeros((n_alpha, n_beta), dtype=int)

    files = sorted(glob.glob(os.path.join(GRID_DIR, "alpha_*.jsonl")))
    if not files:
        raise FileNotFoundError(
            f"{GRID_DIR} 下没有找到 alpha_*.jsonl，先跑 experiments/run_grid.py（本地测试）"
            f"或 slurm/run_grid.slurm（集群）"
        )

    per_cell_vals: dict[tuple[int, int], list[float]] = {}
    for path in files:
        alpha_idx = int(os.path.basename(path).replace("alpha_", "").replace(".jsonl", ""))
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("converged", True):
                    continue
                beta_idx = r["beta_idx"]
                per_cell_vals.setdefault((alpha_idx, beta_idx), []).append(r["delta"])

    for (alpha_idx, beta_idx), vals in per_cell_vals.items():
        if alpha_idx < n_alpha and beta_idx < n_beta:
            delta_mean[alpha_idx, beta_idx] = float(np.mean(vals))
            delta_count[alpha_idx, beta_idx] = len(vals)

    return alphas, betas, delta_mean, delta_count


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_alpha", type=int, default=20)
    parser.add_argument("--n_beta", type=int, default=20)
    parser.add_argument("--alpha_min", type=float, default=0.025)
    parser.add_argument("--alpha_max", type=float, default=0.25)
    parser.add_argument("--beta_min", type=float, default=1e-6)
    parser.add_argument("--beta_max", type=float, default=2e-5)
    args = parser.parse_args()

    alphas, betas, delta_mean, delta_count = load_grid(
        args.n_alpha, args.n_beta, args.alpha_min, args.alpha_max, args.beta_min, args.beta_max
    )

    n_cells_filled = int(np.sum(~np.isnan(delta_mean)))
    n_cells_total = args.n_alpha * args.n_beta
    all_deltas = delta_mean[~np.isnan(delta_mean)]

    fig, ax = plt.subplots(figsize=(6.8, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.pcolormesh(
        betas, alphas, delta_mean, cmap=BLUE_SEQUENTIAL, vmin=0.0, vmax=1.0, shading="nearest",
    )
    ax.set_xscale("log")
    ax.set_xlabel("beta（探索衰减率，对数轴）", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("alpha（学习率）", color=INK_SECONDARY, fontsize=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.set_title(
        f"利润增益 Δ 热力图（{n_cells_filled}/{n_cells_total} 格点已完成）",
        color=INK_PRIMARY, fontsize=11, loc="left",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("平均利润增益 Δ", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)

    fig.tight_layout()
    out_path = os.path.join(BASE, "results", "grid_heatmap.png")
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print(f"saved {out_path}")

    if len(all_deltas) > 0:
        frac_in_paper_range = float(np.mean((all_deltas >= 0.70) & (all_deltas <= 0.90)))
        print(f"已完成格点数: {n_cells_filled}/{n_cells_total}")
        print(f"格点平均 Delta 的均值: {all_deltas.mean():.3f}（论文 Table 1 All 列: {PAPER_TABLE1_ALL_MEAN}）")
        print(f"格点平均 Delta 落在论文 Figure 1 报告的 [0.70, 0.90] 区间的比例: {frac_in_paper_range:.1%}")
        print(f"单格点最少/最多 session 数: {delta_count[delta_count > 0].min()}/{delta_count.max()}")
    else:
        print("还没有任何格点完成，先跑一些 session。")


if __name__ == "__main__":
    main()
