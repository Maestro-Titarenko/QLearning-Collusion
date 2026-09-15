"""
Reproduces the paper's Figure 4: how deviator vs. non-deviator prices evolve
over time after an exogenous deviation. Takes the price path from each
session produced by experiments/run_anatomy.py (already averaged within the
session over cycle starting points and deviator identity) and averages once
more across sessions.
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


def main() -> None:
    with open(os.path.join(BASE, "results", "anatomy_experiment.json")) as f:
        data = json.load(f)

    grid0 = np.array(data["grid0"])
    p_nash = data["p_nash"][0]
    p_monopoly = data["p_monopoly"][0]

    deviator_idx_paths = np.array([s["deviator_path"] for s in data["sessions"]])  # (n_sessions, horizon+1)
    nondeviator_idx_paths = np.array([s["nondeviator_path"] for s in data["sessions"]])

    # The index paths are "averaged fractional indices" (since the session
    # already averaged over multiple starting points), so they're mapped
    # back to actual prices via linear interpolation on the price grid,
    # rather than treated as integer indices for a direct table lookup.
    def idx_to_price(idx_array):
        return np.interp(idx_array, np.arange(len(grid0)), grid0)

    deviator_price_paths = idx_to_price(deviator_idx_paths)
    nondeviator_price_paths = idx_to_price(nondeviator_idx_paths)

    deviator_avg = deviator_price_paths.mean(axis=0)
    nondeviator_avg = nondeviator_price_paths.mean(axis=0)
    long_run_price = (deviator_avg[-1] + nondeviator_avg[-1]) / 2

    taus = np.arange(len(deviator_avg))

    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(taus, deviator_avg, color=ORANGE, linewidth=1.8, marker="o", markersize=4, label="Deviating agent")
    ax.plot(taus, nondeviator_avg, color=BLUE, linewidth=1.8, marker="^", markersize=4, label="Nondeviating agent")
    ax.axhline(p_nash, color=INK_MUTED, linewidth=1.0, linestyle=":")
    ax.axhline(p_monopoly, color=INK_MUTED, linewidth=1.0, linestyle="-.")
    ax.axhline(long_run_price, color=INK_MUTED, linewidth=1.0, linestyle="-", alpha=0.6)

    ax.text(taus[-1], p_nash, "  Nash", color=INK_SECONDARY, fontsize=9, va="center")
    ax.text(taus[-1], p_monopoly, "  Monopoly", color=INK_SECONDARY, fontsize=9, va="center")
    ax.text(taus[-1], long_run_price, "  Long-run price", color=INK_SECONDARY, fontsize=9, va="bottom")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel("τ (the deviation happens at τ=1)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Price", color=INK_SECONDARY, fontsize=10)
    ax.set_title(f"Impulse response after an exogenous deviation\n(average over {len(data['sessions'])} converged sessions, cf. paper's Figure 4)",
                 color=INK_PRIMARY, fontsize=10.5, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.02, 0.85))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    out_path = os.path.join(BASE, "results", "anatomy_impulse_response.png")
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
