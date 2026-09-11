"""
复现论文 Figure 4：外生偏离后，偏离方 vs 非偏离方的价格如何随时间演化。
把 experiments/run_anatomy.py 产出的每个 session 的（已经在 session 内对循环
起点和偏离方身份取过平均的）价格路径，再取一次跨 session 的平均。
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def main() -> None:
    with open(os.path.join(BASE, "results", "anatomy_experiment.json")) as f:
        data = json.load(f)

    grid0 = np.array(data["grid0"])
    p_nash = data["p_nash"][0]
    p_monopoly = data["p_monopoly"][0]

    deviator_idx_paths = np.array([s["deviator_path"] for s in data["sessions"]])  # (n_sessions, horizon+1)
    nondeviator_idx_paths = np.array([s["nondeviator_path"] for s in data["sessions"]])

    # 索引路径是"平均后的分数索引"（因为 session 内已经对多个起点取过平均），
    # 用价格网格做线性插值映射回实际价格，而不是直接当整数索引查表。
    def idx_to_price(idx_array):
        return np.interp(idx_array, np.arange(len(grid0)), grid0)

    deviator_price_paths = idx_to_price(deviator_idx_paths)
    nondeviator_price_paths = idx_to_price(nondeviator_idx_paths)

    deviator_avg = deviator_price_paths.mean(axis=0)
    nondeviator_avg = nondeviator_price_paths.mean(axis=0)
    long_run_price = (deviator_avg[-1] + nondeviator_avg[-1]) / 2

    taus = np.arange(len(deviator_avg))

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(taus, deviator_avg, color=ORANGE, linewidth=1.8, marker="o", markersize=4, label="偏离方 (deviating agent)")
    ax.plot(taus, nondeviator_avg, color=BLUE, linewidth=1.8, marker="^", markersize=4, label="非偏离方 (nondeviating agent)")
    ax.axhline(p_nash, color=INK_MUTED, linewidth=1.0, linestyle=":")
    ax.axhline(p_monopoly, color=INK_MUTED, linewidth=1.0, linestyle="-.")
    ax.axhline(long_run_price, color=INK_MUTED, linewidth=1.0, linestyle="-", alpha=0.6)

    ax.text(taus[-1], p_nash, "  Nash", color=INK_SECONDARY, fontsize=9, va="center")
    ax.text(taus[-1], p_monopoly, "  Monopoly", color=INK_SECONDARY, fontsize=9, va="center")
    ax.text(taus[-1], long_run_price, "  长期价格", color=INK_SECONDARY, fontsize=9, va="bottom")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel("τ（偏离发生在 τ=1）", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("价格", color=INK_SECONDARY, fontsize=10)
    ax.set_title(f"外生偏离后的脉冲响应（{len(data['sessions'])} 个收敛 session 的平均，对应论文 Figure 4）",
                 color=INK_PRIMARY, fontsize=11, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.02, 0.85))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    out_path = os.path.join(BASE, "results", "anatomy_impulse_response.png")
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
