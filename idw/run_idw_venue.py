"""Test the IDW model on the concert venue and stadium sound maps.

Run from the repo root:  python3 idw/run_idw_venue.py
Prints RMSE  for each phone count and layout, and saves truth / prediction / error maps for each venue.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Working_final_maps"))
OUT_DIR = HERE                                  # saving figures next to this script

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np

import concert_map
import stadium_map
from extract_sound_venues import prepare_interpolation_data
from idw_model import idw, choose_p_k

VENUES = {
    "concert": concert_map,
    "stadium": stadium_map,
}
FRAME = {"concert": 0, "stadium": 60}           # time step shown in the figure (stadium: 60 = goal at 120 s)
phone_counts = [10, 30, 50, 100, 250]
layouts = ["random", "grid", "clustered"]
SEED = 42
FIGURE_SETUP = (100, "grid")

def rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))


def to_grid(values, mask):
    """Put audience-area values back onto the full map grid (NaN elsewhere)."""
    grid = np.full(mask.shape, np.nan)
    grid[mask] = values
    return grid


for name, venue in VENUES.items():
    print(f"\n=== {name} === (loading sound map...)")
    sound_maps, x, y, allowed_mask = venue.load_sound_map()

    prepared = prepare_interpolation_data(sound_maps, x, y, phone_counts, layouts,
                                          seed=SEED, allowed_mask=allowed_mask)

    for count in phone_counts:
        for layout in layouts:
            data = prepared[(count, layout)]
            pos, lev, q, true = (data["phone_positions"], data["phone_levels"],
                                 data["query_positions"], data["true_levels"])

            #errors only measured in the audience area (query_positions = allowed area)
            base_rmse = rmse(idw(pos, lev, q), true)
            p, k = choose_p_k(pos, lev)
            tune_rmse = rmse(idw(pos, lev, q, p=p, k=k), true)
            
            print(f"Phones: {count:3d} | Layout: {layout:10s} | "
                  f"default RMSE: {base_rmse:5.2f} dB | tuned RMSE: {tune_rmse:5.2f} dB "
                  f"(p={p:g}, k={'all' if k is None else k})")

    # Figure: truth, tuned prediction and error for one configuration and time step
    data = prepared[FIGURE_SETUP]
    pos, lev, q, true, mask = (data["phone_positions"], data["phone_levels"],
                               data["query_positions"], data["true_levels"],
                               data["allowed_mask"])
    p, k = choose_p_k(pos, lev)
    pred = idw(pos, lev, q, p=p, k=k)

    t = min(FRAME[name], true.shape[0] - 1)
    true_grid = to_grid(true[t], mask)
    pred_grid = to_grid(pred[t], mask)
    err_grid = pred_grid - true_grid
    vmin, vmax = np.nanmin(true_grid), np.nanmax(true_grid)
    elim = np.nanmax(np.abs(err_grid))

    n, layout = FIGURE_SETUP
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    panels = [(true_grid, f"Ground truth (frame {t})", "viridis", vmin, vmax),
              (pred_grid, f"IDW prediction ({n} phones, {layout}, p={p:g}, k={'all' if k is None else k})",
               "viridis", vmin, vmax),
              (err_grid, "Error (prediction - truth)", "RdBu_r", -elim, elim)]
    for ax, (img, title, cmap, lo, hi) in zip(axes, panels):
        im = ax.pcolormesh(x, y, img, shading="auto", cmap=cmap, vmin=lo, vmax=hi)
        venue.draw_venue(ax)
        ax.scatter(pos[:, 0], pos[:, 1], c="red", s=10, edgecolors="white",
                   linewidths=0.4, zorder=10)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, label="dB", shrink=0.8)
    fig.suptitle(f"IDW on the {name} map")
    plt.savefig(os.path.join(OUT_DIR, f"idw_venue_{name}.png"), dpi=150)
    plt.close(fig)
    print(f"Saved idw_venue_{name}.png")
