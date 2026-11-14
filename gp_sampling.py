"""Prepare shared single-source samples for GP and Kriging.

Run this file first. It creates observations and query points, but it does not
run either interpolation model. Files inside evaluation_only are for checking
the simulation and must not be given to a prediction model.
"""

from pathlib import Path
import csv
import sys

import matplotlib

matplotlib.use("Agg")  # Save the figure; no pop-up window is needed.
import matplotlib.pyplot as plt
import numpy as np


# Andrei's folder is next to austin/. Use his existing field and sampler.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "andreib"))
import point_source
from extract_sound import extract_measurements, prepare_interpolation_data


# Change these three settings to prepare another experiment. The same seed
# reproduces each configuration, but 20 random phones are not necessarily a
# subset of the 50 or 100 random phones.
PHONE_COUNTS = (5, 10, 20, 50, 100)
LAYOUTS = ("random", "grid", "clustered")
SEED = 42


def save_observations(path, positions, levels):
    """Save one row per phone: x position, y position, observed level."""
    table = np.column_stack((positions, levels))
    np.savetxt(
        path, table, delimiter=",", fmt="%.10f", comments="",
        header="x_m,y_m,observed_level_db",
    )


def main():
    output = HERE / "gp_single_source_sampling_output"
    inputs = output / "model_inputs"
    evaluation = output / "evaluation_only"
    inputs.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)

    # Andrei's function generates every requested configuration from one scene.
    data = prepare_interpolation_data(
        point_source.sound_maps, point_source.x, point_source.y,
        phone_counts=PHONE_COUNTS, layouts=LAYOUTS, seed=SEED,
    )

    # Every configuration predicts on this same 201 x 201 grid.
    first = data[(PHONE_COUNTS[0], LAYOUTS[0])]
    query_positions = first["query_positions"]  # (40401, 2)
    true_levels = first["true_levels"][0]        # (40401,)
    np.savetxt(
        inputs / "query_points.csv", query_positions, delimiter=",",
        fmt="%.10f", comments="", header="x_m,y_m",
    )
    np.savetxt(
        evaluation / "truth_grid.csv",
        np.column_stack((query_positions, true_levels)), delimiter=",",
        fmt="%.10f", comments="", header="x_m,y_m,true_level_db",
    )

    # The source position and true peak are used only in the diagnostic summary.
    source = np.array([
        point_source.source_x_positions[0], point_source.source_y[0]
    ])
    summary_path = evaluation / "sampling_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "phone_count", "layout", "seed", "nearest_source_distance_m",
            "max_observed_level_db", "true_peak_db",
        ])

        fig, axes = plt.subplots(
            len(LAYOUTS), len(PHONE_COUNTS),
            figsize=(3.2 * len(PHONE_COUNTS) + 1, 3 * len(LAYOUTS)),
            sharex=True, sharey=True, constrained_layout=True, squeeze=False,
        )
        for row, layout in enumerate(LAYOUTS):
            for col, count in enumerate(PHONE_COUNTS):
                sample = data[(count, layout)]
                positions = sample["phone_positions"]       # (count, 2)
                levels = sample["phone_levels"][0]          # (count,)
                assert np.array_equal(sample["query_positions"], query_positions)
                assert len(positions) == len(levels) == count

                # This CSV is the only configuration-specific input for models.
                save_observations(
                    inputs / f"observations_{count}_{layout}.csv",
                    positions, levels,
                )

                distances = np.linalg.norm(positions - source, axis=1)
                nearest = float(distances.min())
                writer.writerow([
                    count, layout, SEED, f"{nearest:.6f}",
                    f"{levels.max():.6f}", f"{true_levels.max():.6f}",
                ])
                print(f"{count:3d} {layout:9s}: nearest phone {nearest:5.2f} m, "
                      f"highest reading {levels.max():6.2f} dB")

                # Red star is the known source, shown only in this diagnostic figure.
                ax = axes[row, col]
                dots = ax.scatter(
                    positions[:, 0], positions[:, 1], c=levels,
                    vmin=true_levels.min(), vmax=true_levels.max(),
                    cmap="viridis", s=27,
                )
                ax.scatter(source[0], source[1], color="red", marker="*", s=130)
                ax.set(title=f"{count} phones, {layout}", xlim=(0, 100), ylim=(0, 100))
                ax.set_aspect("equal")
                if row == len(LAYOUTS) - 1:
                    ax.set_xlabel("x (m)")
                if col == 0:
                    ax.set_ylabel("y (m)")

    fig.colorbar(dots, ax=axes, label="Observed level (dB)", shrink=0.8)
    fig.savefig(evaluation / "sampling_layouts.png", dpi=160)
    plt.close("all")

    # One extra source-centre observation is a controlled diagnostic only.
    # It must not replace the original 20-random-phone benchmark.
    if (20, "random") in data:
        baseline = data[(20, "random")]
        diagnostic_positions = np.vstack((baseline["phone_positions"], source))
        diagnostic_levels = extract_measurements(
            point_source.sound_maps, point_source.x, point_source.y,
            diagnostic_positions,
        )[0]
        save_observations(
            evaluation / "diagnostic_20_random_plus_source.csv",
            diagnostic_positions, diagnostic_levels,
        )

    print("\nModel inputs:", inputs)
    print("Evaluation and diagnostic files:", evaluation)
    if (20, "random") in data:
        print("The extra source-centre point is for diagnosis, not fair method scoring.")


if __name__ == "__main__":
    main()
