"""
Reproduces the paper's Section V.A: more firms means lower Delta. Uses the
140 sessions from Phase 2's representative-experiment run for n=2, and this
phase's own runs for n=3/4.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

PAPER_DELTA = {2: 0.849, 3: 0.64, 4: 0.56}  # numbers from the paper's main text (n=3/4 use the baseline beta grid, not a beta tuned down specially)


def load_deltas(path: str) -> np.ndarray:
    rows = [json.loads(l) for l in open(path)]
    return np.array([r["delta"] for r in rows if r.get("converged", True)])


def main() -> None:
    n2 = load_deltas(os.path.join(BASE, "results", "representative_experiment.jsonl"))
    n3 = load_deltas(os.path.join(BASE, "results", "n3_experiment.jsonl"))
    n4 = load_deltas(os.path.join(BASE, "results", "n4_experiment.jsonl"))

    ns = [2, 3, 4]
    means = [n2.mean(), n3.mean(), n4.mean()]
    ses = [n2.std() / np.sqrt(len(n2)), n3.std() / np.sqrt(len(n3)), n4.std() / np.sqrt(len(n4))]
    ns_counts = [len(n2), len(n3), len(n4)]
    paper_vals = [PAPER_DELTA[n] for n in ns]

    fig, ax = plt.subplots(figsize=(6.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    width = 0.32
    x = np.arange(len(ns))
    ax.bar(x - width / 2, means, width=width, yerr=ses, color=BLUE, capsize=4, label="Replicated")
    ax.bar(x + width / 2, paper_vals, width=width, color=ORANGE, alpha=0.85, label="Paper's main text")

    for i, (m, cnt) in enumerate(zip(means, ns_counts)):
        ax.text(x[i] - width / 2, m + 0.02, f"{m:.2f}\n(n_sessions={cnt})", ha="center", fontsize=8, color=INK_SECONDARY)
    for i, p in enumerate(paper_vals):
        ax.text(x[i] + width / 2, p + 0.02, f"{p:.2f}", ha="center", fontsize=8, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in ns], color=INK_PRIMARY, fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_ylabel("Mean profit gain Δ", color=INK_SECONDARY, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title("More firms means less collusion (but it doesn't disappear)", color=INK_PRIMARY, fontsize=11, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in legend.get_texts():
        t.set_color(INK_SECONDARY)

    fig.tight_layout()
    out_path = os.path.join(BASE, "results", "n_players_comparison.png")
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print(f"saved {out_path}")

    print(f"n=2: mean={means[0]:.3f} (n_sessions={ns_counts[0]}, paper={paper_vals[0]})")
    print(f"n=3: mean={means[1]:.3f} (n_sessions={ns_counts[1]}, paper={paper_vals[1]})")
    print(f"n=4: mean={means[2]:.3f} (n_sessions={ns_counts[2]}, paper={paper_vals[2]})")


if __name__ == "__main__":
    main()
