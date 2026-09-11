"""
画两张图：
  左：单个代表性 session 的学习曲线（窗口平均利润增益 Delta_t 随训练期数的
      变化），对应论文 Figure 10 的风格。
  右：140 个 session 的 Delta 分布直方图，标出我们复现的均值和论文 Table 1
      "All" 列报告的均值 0.849，方便直接对比。
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 配色取自 dataviz 技能的参考色板（浅色模式）
BLUE = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def main() -> None:
    with open(os.path.join(BASE, "results", "sample_trace.json")) as f:
        trace = json.load(f)
    rows = [json.loads(l) for l in open(os.path.join(BASE, "results", "representative_experiment.jsonl"))]
    deltas = np.array([r["delta"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

    # ---- 左图：学习曲线 ----
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    periods = np.array(trace["trace_periods"])
    avg_profit = np.array(trace["trace_avg_profit"])
    pi_nash, pi_monopoly = trace["pi_nash"], trace["pi_monopoly"]
    delta_t = (avg_profit - pi_nash) / (pi_monopoly - pi_nash)

    ax.plot(periods, delta_t, color=BLUE, linewidth=1.6, solid_capstyle="round")
    ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax.text(periods[-1], 0.02, "Nash (Δ=0)", color=INK_SECONDARY, fontsize=9, ha="right", va="bottom")
    ax.text(periods[-1], 0.98, "Monopoly (Δ=1)", color=INK_SECONDARY, fontsize=9, ha="right", va="top")
    ax.axvline(trace["n_periods"], color=INK_MUTED, linewidth=0.8, linestyle=":")
    ax.text(trace["n_periods"], -0.12, "收敛", color=INK_SECONDARY, fontsize=8, ha="center")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel("训练期数 t", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("窗口平均利润增益 Δ_t", color=INK_SECONDARY, fontsize=9)
    ax.set_title(f"代表性 session 的学习曲线（seed=2024，最终 Δ={trace['delta']:.2f}，极限环长度={trace['cycle_length']}）",
                 color=INK_PRIMARY, fontsize=10, loc="left")

    # ---- 右图：Delta 分布 ----
    ax2 = axes[1]
    ax2.set_facecolor(SURFACE)
    ax2.hist(deltas, bins=18, color=BLUE, alpha=0.85, edgecolor=SURFACE, linewidth=0.5)
    ax2.axvline(deltas.mean(), color=INK_PRIMARY, linewidth=1.6)
    ax2.axvline(0.849, color="#eb6834", linewidth=1.6, linestyle="--")
    ax2.text(deltas.mean(), ax2.get_ylim()[1] if False else 0, "", alpha=0)  # placeholder to keep autoscale stable

    ymax = ax2.get_ylim()[1]
    ax2.text(deltas.mean(), ymax * 0.95, f"复现均值 {deltas.mean():.3f}", color=INK_PRIMARY, fontsize=9,
              ha="left" if deltas.mean() < 0.849 else "right", va="top")
    ax2.text(0.849, ymax * 0.80, "论文 Table 1: 0.849", color="#eb6834", fontsize=9,
              ha="right" if deltas.mean() < 0.849 else "left", va="top")

    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax2.spines[spine].set_color(GRID)
    ax2.tick_params(colors=INK_MUTED, labelsize=8)
    ax2.grid(axis="y", color=GRID, linewidth=0.8)
    ax2.set_xlabel("利润增益 Δ（每 session）", color=INK_SECONDARY, fontsize=9)
    ax2.set_ylabel("session 数", color=INK_SECONDARY, fontsize=9)
    ax2.set_title(f"140 个 session 的 Δ 分布（均值 {deltas.mean():.3f}，标准差 {deltas.std():.3f}）",
                  color=INK_PRIMARY, fontsize=10, loc="left")

    fig.tight_layout()
    out_path = os.path.join(BASE, "results", "representative_experiment.png")
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
