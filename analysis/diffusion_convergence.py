"""Does diffusion LMS get worse with more phones because it hasn't converged?

Runs diffusion LMS on grid layouts (perfect phone readings) with more and more
iterations per time step. If the error keeps dropping with more iterations at high
phone counts, the default 900 iterations is too few for large networks.

    python3 analysis/diffusion_convergence.py
Writes analysis/results/diffusion_convergence.csv and figures/diffusion_convergence.pdf/.png
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import compare_models as cm                      # sets up the import paths
from bias_and_variance_extract_sound_venues import prepare_interpolation_data
from diffusion_lms_model import DiffusionAcousticLMS
import importlib

RESULTS = ROOT / "analysis" / "results"
ITERATIONS = [300, 900, 3000, 9000]
COUNTS = [30, 100, 250]


def main():
    rows = []
    for venue in ["concert", "f1"]:
        module, stride = cm.VENUES[venue]
        maps, x, y, mask = importlib.import_module(module).load_sound_map()
        maps = np.asarray(maps)
        maps = (maps[None] if maps.ndim == 2 else maps)[::stride]
        data = prepare_interpolation_data(maps, x, y, COUNTS, ["grid"], seed=0,
                                          allowed_mask=mask, phone_error="perfect")
        for (count, layout), d in data.items():
            for n_iter in ITERATIONS:
                settings = dict(cm.DIFF_SETTINGS, n_iterations=n_iter)
                t0 = time.perf_counter()
                pred = DiffusionAcousticLMS(**settings).fit_predict(
                    d["phone_positions"], d["phone_levels"], d["query_positions"])
                s = cm.score(pred, d["true_levels"], d["evaluation_mask"])
                rows.append(dict(venue=venue, phone_count=count, n_iterations=n_iter,
                                 runtime_s=time.perf_counter() - t0, **s))
                print(f"{venue:8s} {count:3d} phones, {n_iter:5d} iterations: "
                      f"RMSE {s['rmse_db']:.2f} dB", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "diffusion_convergence.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
    colours = {30: "#2a78d6", 100: "#eb6834", 250: "#1baf7a"}
    for ax, (venue, title) in zip(axes, [("concert", "Concert"), ("f1", "F1 racetrack")]):
        for count in COUNTS:
            d = df[(df.venue == venue) & (df.phone_count == count)]
            ax.plot(d.n_iterations, d.rmse_db, "-o", color=colours[count], linewidth=1.6,
                    markersize=4.5, label=f"{count} phones (grid)")
        ax.axvline(900, color="#6b6b6b", linewidth=0.8, linestyle=":")
        ax.text(900, ax.get_ylim()[1], " default", fontsize=7, color="#6b6b6b", va="top")
        ax.set_xscale("log")
        ax.set_xticks(ITERATIONS)
        ax.set_xticklabels([str(i) for i in ITERATIONS])
        ax.minorticks_off()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Iterations per time step")
        ax.grid(axis="y", color="#e4e4e0", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("RMSE (dB)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figs = RESULTS / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figs / f"diffusion_convergence.{ext}", dpi=200)
    print("Saved", RESULTS / "diffusion_convergence.csv")


if __name__ == "__main__":
    main()
