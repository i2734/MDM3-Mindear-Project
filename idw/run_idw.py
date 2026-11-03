import os
import sys

# finding the group files (extract_sound.py, scenarios) in the parent folder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))   # save figures next to this script

import matplotlib
matplotlib.use("Agg")                      # no pop-up windows
import matplotlib.pyplot as plt
import numpy as np

import point_source
import multiple_point_sources
import single_moving_point_source
from extract_sound import prepare_interpolation_data
from idw_model import idw

SCENARIOS = {"point_source": point_source,
    "multiple_point_sources": multiple_point_sources,
    "single_moving_point_source": single_moving_point_source,
}
phone_counts = [5, 10, 20, 50, 100]
layouts = ["random", "grid", "clustered"]
SEEDS = [42]


for name, scenario in SCENARIOS.items():
    print(f"\n=== {name} ===")
    x, y = scenario.x, scenario.y
    # preparing interpolation data for each seed, phone count, and layout
    prepared = {s: prepare_interpolation_data(scenario.sound_maps, x, y,
                                              phone_counts, layouts, seed=s)
                for s in SEEDS}
    # evaluating IDW for each scenario, phone count, layout, and seed
    for count in phone_counts:
        for layout in layouts:
            rmse, mae = [], []
            for s in SEEDS:
                data = prepared[s][(count, layout)]
                pred = idw(data["phone_positions"], data["phone_levels"],
                           data["query_positions"])
                diff = pred - data["true_levels"]
                rmse.append(np.sqrt(np.mean(diff ** 2)))
                mae.append(np.mean(np.abs(diff)))
            print(f"Phones: {count:3d} | Layout: {layout:10s} | "
                  f"RMSE: {np.mean(rmse):.2f} dB | MAE: {np.mean(mae):.2f} dB")

    # Figure: truth vs prediction (100 grid phones, time step 0)
    data = prepared[SEEDS[0]][(100, "grid")]
    pred = idw(data["phone_positions"], data["phone_levels"], data["query_positions"])
    shape = (len(y), len(x))
    extent = [x.min(), x.max(), y.min(), y.max()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(data["true_levels"][0].reshape(shape), origin="lower", extent=extent)
    axes[0].set_title("Ground Truth (t=0)")
    plt.colorbar(im0, ax=axes[0], label="dB")
    im1 = axes[1].imshow(pred[0].reshape(shape), origin="lower", extent=extent)
    axes[1].scatter(data["phone_positions"][:, 0], data["phone_positions"][:, 1],
                    color="red", s=20, label="Phones")
    axes[1].set_title("IDW Prediction (100 phones, grid)")
    axes[1].legend()
    plt.colorbar(im1, ax=axes[1], label="dB")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"idw_{name}.png"), dpi=150)
    plt.close(fig)
