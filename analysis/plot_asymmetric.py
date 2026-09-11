"""
复现论文 Table 4（成本不对称）的图示版本：Delta 和 2 号企业的静态 Nash 市场
份额随 c2 的变化，跟论文数字对照。
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

PAPER_DELTA = {1.0: 0.849, 0.875: 0.841, 0.75: 0.812, 0.625: 0.781, 0.5: 0.759, 0.25: 0.713}
PAPER_SHARE = {1.0: 0.500, 0.875: 0.545, 0.75: 0.588, 0.625: 0.627, 0.5: 0.662, 0.25: 0.722}


def main() -> None:
    with open(os.path.join(BASE, "results", "asymmetric_experiment.json")) as f:
        data = json.load(f)

    c2_values = sorted((float(k) for k in data.keys()), reverse=True)
    deltas_mean, deltas_se, shares = [], [], []
    for c2 in c2_values:
        r = data[str(c2)]
        d = np.array([s["delta"] for s in r["sessions"]])
        deltas_mean.append(d.mean())
        deltas_se.append(d.std() / np.sqrt(len(d)))
        shares.append(r["share2_nash"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.errorbar(c2_values, deltas_mean, yerr=deltas_se, color=BLUE, linewidth=1.8, marker="o", markersize=5,
                capsize=3, label="复现（15 session/点）")
    ax.plot(c2_values, [PAPER_DELTA[c2] for c2 in c2_values], color=ORANGE, linewidth=1.8, marker="s", markersize=5,
            linestyle="--", label="论文 Table 4")
    ax.invert_xaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel("c2（企业 2 的边际成本，c1=1 固定）", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("平均利润增益 Δ", color=INK_SECONDARY, fontsize=10)
    ax.set_title("成本不对称越大，Δ 略微下降", color=INK_PRIMARY, fontsize=11, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for t in legend.get_texts():
        t.set_color(INK_SECONDARY)

    ax2 = axes[1]
    ax2.set_facecolor(SURFACE)
    ax2.plot(c2_values, shares, color=BLUE, linewidth=1.8, marker="o", markersize=5, label="复现（解析解）")
    ax2.plot(c2_values, [PAPER_SHARE[c2] for c2 in c2_values], color=ORANGE, linewidth=1.2, marker="s", markersize=4,
             linestyle="--", label="论文 Table 4")
    ax2.invert_xaxis()
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax2.spines[spine].set_color(GRID)
    ax2.tick_params(colors=INK_MUTED, labelsize=9)
    ax2.grid(axis="y", color=GRID, linewidth=0.8)
    ax2.set_xlabel("c2（企业 2 的边际成本，c1=1 固定）", color=INK_SECONDARY, fontsize=10)
    ax2.set_ylabel("企业 2 的静态 Nash 市场份额", color=INK_SECONDARY, fontsize=10)
    ax2.set_title("市场份额（解析解，两条线完全重合）", color=INK_PRIMARY, fontsize=11, loc="left")
    legend2 = ax2.legend(frameon=False, fontsize=9, loc="upper left")
    for t in legend2.get_texts():
        t.set_color(INK_SECONDARY)

    fig.tight_layout()
    out_path = os.path.join(BASE, "results", "asymmetric_experiment.png")
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
