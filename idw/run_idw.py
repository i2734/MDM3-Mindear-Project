import os
import sys

#finding the group files (extract_sound.py, scenarios) in the parent folder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))   # save figures next to this script

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import point_source
import multiple_point_sources
import single_moving_point_source
from extract_sound import prepare_interpolation_data
from idw_model import idw, choose_p_k

SCENARIOS = {
    "point_source": point_source,
    "multiple_point_sources": multiple_point_sources,
    "single_moving_point_source": single_moving_point_source,
}
phone_counts = [5, 10, 20, 50, 100]
layouts = ["random", "grid", "clustered"]
SEEDS = [42]

def errors(pred, true):
    diff = pred - true
    return np.sqrt(np.mean(diff ** 2)), np.mean(np.abs(diff))


for name, scenario in SCENARIOS.items():
    print(f"\n=== {name} ===")
    print("                                 default (p=2, all phones)  |  tuned (chosen from phones only)")
    x, y = scenario.x, scenario.y

    prepared = {s: prepare_interpolation_data(scenario.sound_maps, x, y,
                                              phone_counts, layouts, seed=s)
                for s in SEEDS}

    for count in phone_counts:
        for layout in layouts:
            base_r, base_m, tune_r, tune_m, chosen = [], [], [], [], []
            for s in SEEDS:
                data = prepared[s][(count, layout)]
                pos, lev, q, true = (data["phone_positions"], data["phone_levels"],
                                     data["query_positions"], data["true_levels"])

                # default model
                r, m = errors(idw(pos, lev, q), true)
                base_r.append(r); base_m.append(m)

                # improved model: p and k chosen by leaving one phone out to see which values give the best prediction for that phone
                p, k = choose_p_k(pos, lev)
                r, m = errors(idw(pos, lev, q, p=p, k=k), true)
                tune_r.append(r); tune_m.append(m); chosen.append((p, k))

            p, k = chosen[0]
            print(f"Phones: {count:3d} | Layout: {layout:10s} | "
                  f"RMSE: {np.mean(base_r):5.2f} dB | MAE: {np.mean(base_m):5.2f} dB  |  "
                  f"RMSE: {np.mean(tune_r):5.2f} dB | MAE: {np.mean(tune_m):5.2f} dB "
                  f"(p={p:g}, k={'all' if k is None else k})")

    # figure: truth, tuned prediction and error, on the same colour scale
    data = prepared[SEEDS[0]][(100, "grid")]
    pos, lev, q, true = (data["phone_positions"], data["phone_levels"],
                         data["query_positions"], data["true_levels"])
    p, k = choose_p_k(pos, lev)
    pred = idw(pos, lev, q, p=p, k=k)

    shape = (len(y), len(x))
    extent = [x.min(), x.max(), y.min(), y.max()]
    true_map, pred_map = true[0].reshape(shape), pred[0].reshape(shape)
    err_map = pred_map - true_map
    vmin, vmax = true_map.min(), true_map.max()
    elim = np.abs(err_map).max()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    panels = [(true_map, "Ground Truth (t=0)", "viridis", vmin, vmax, "dB"),
              (pred_map, f"IDW Prediction (100 phones, grid, p={p:g}, k={'all' if k is None else k})",
               "viridis", vmin, vmax, "dB"),
              (err_map, "Error (prediction - truth)", "RdBu_r", -elim, elim, "dB")]
    for ax, (img, title, cmap, lo, hi, label) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi)
        ax.scatter(pos[:, 0], pos[:, 1], color="red", s=12)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, label=label)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"idw_{name}.png"), dpi=150)
    plt.close(fig)
