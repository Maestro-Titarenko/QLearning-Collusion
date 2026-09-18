"""
Reproduces the paper's Figure 1: a heatmap of the mean profit gain Delta over
each cell of the (alpha, beta) grid. Reads results/grid/alpha_*.jsonl
produced by experiments/run_grid.py, averages sessions by
(alpha_idx, beta_idx), and plots the heatmap.

Delta is a continuous quantity (the paper's Section 9, eq. 9), plotted with a
single-hue sequential color scale (light->dark blue, from the dataviz
skill's reference palette's blue sequential ramp), not a categorical color —
so this shouldn't use a default rainbow colormap like tab10/viridis.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_DIR = os.path.join(BASE, "results", "grid")

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_LINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# dataviz skill reference palette: blue sequential ramp, 100 (light) -> 700
# (dark), with light representing "close to 0".
BLUE_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_sequential",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

PAPER_TABLE1_ALL_MEAN = 0.849  # the paper's Table 1 "All" column, the full-grid average


def load_grid(
    n_alpha: int, n_beta: int, alpha_min: float, alpha_max: float, beta_min: float, beta_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (alphas, betas, delta_mean, n_sessions_per_cell).

    alphas/betas are computed directly from the same formula
    (linspace/geomspace) used in experiments/run_grid.py, rather than
    backfilled from the data files — if every session at some cell failed to
    converge, there's no usable alpha/beta value in the data file, and
    backfilling would leave that cell's coordinates missing.
    """
    alphas = np.linspace(alpha_min, alpha_max, n_alpha)
    betas = np.geomspace(beta_min, beta_max, n_beta)

    delta_mean = np.full((n_alpha, n_beta), np.nan)
    delta_count = np.zeros((n_alpha, n_beta), dtype=int)

    files = sorted(glob.glob(os.path.join(GRID_DIR, "alpha_*.jsonl")))
    if not files:
        raise FileNotFoundError(
            f"No alpha_*.jsonl found under {GRID_DIR}; run experiments/run_grid.py "
            f"(local test) or slurm/run_grid.slurm (cluster) first"
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

    # A fixed 0-1 scale is the theoretically meaningful range (0=competitive,
    # 1=full collusion), but the actual cell means land in a much narrower
    # band (empirically ~0.62-0.95 here) -- stretching the colormap over the
    # full 0-1 range crushes almost all of that real variation into the top
    # third of the color scale and makes the grid look artificially uniform.
    # Scaling to the 1st/99th percentile of the actual data instead spends
    # the whole color range on the variation that's actually present.
    if len(all_deltas) > 0:
        vmin, vmax = np.percentile(all_deltas, [1, 99])
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(6.8, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.pcolormesh(
        betas, alphas, delta_mean, cmap=BLUE_SEQUENTIAL, vmin=vmin, vmax=vmax, shading="nearest",
    )
    ax.set_xscale("log")
    ax.set_xlabel("beta (exploration decay rate, log scale)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("alpha (learning rate)", color=INK_SECONDARY, fontsize=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.set_title(
        f"Profit gain Δ heatmap ({n_cells_filled}/{n_cells_total} cells completed)",
        color=INK_PRIMARY, fontsize=11, loc="left",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Mean profit gain Δ (color scaled to {vmin:.2f}-{vmax:.2f}, the actual data range)", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)

    fig.tight_layout()
    out_path = os.path.join(BASE, "results", "grid_heatmap.png")
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print(f"saved {out_path}")

    if len(all_deltas) > 0:
        frac_in_paper_range = float(np.mean((all_deltas >= 0.70) & (all_deltas <= 0.90)))
        print(f"Cells completed: {n_cells_filled}/{n_cells_total}")
        print(f"Mean Delta over completed cells: {all_deltas.mean():.3f} (paper's Table 1 All column: {PAPER_TABLE1_ALL_MEAN})")
        print(f"Share of completed cells falling in the paper's reported Figure 1 range of [0.70, 0.90]: {frac_in_paper_range:.1%}")
        print(f"Min/max sessions per completed cell: {delta_count[delta_count > 0].min()}/{delta_count.max()}")
    else:
        print("No cells completed yet; run some sessions first.")


if __name__ == "__main__":
    main()
