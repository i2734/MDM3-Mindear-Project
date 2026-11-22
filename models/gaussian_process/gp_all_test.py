"""Part 3: test the same GP on single, multiple, and moving sources.

Run the sampling and model scripts first. This is the only part that opens
evaluation_only/true_levels.npy and source_positions.npy. It writes:
    overall_metrics.csv       one row per scenario/count/layout
    frame_metrics.csv         one row per time frame
    per_source_metrics.csv    one row per source and time frame
    three_scenario_comparison.csv
    three frame-0, 100-grid sampling figures
    one 3 x 3 truth/prediction/error figure
    one RMSE line figure comparing the three sound fields
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "gp_all_sources_output"
MODEL_INPUTS = OUTPUT / "model_inputs"
PREDICTIONS = OUTPUT / "gp_predictions"
EVALUATION_ONLY = OUTPUT / "evaluation_only"
SOURCE_RADIUS_M = 5.0  # Chosen before scoring, for all three scenarios.
SCENARIOS = ("single_static", "multiple_static", "single_moving")
LABELS = {
    "single_static": "Single stationary source",
    "multiple_static": "Three stationary sources",
    "single_moving": "One moving source",
}


def write_rows(path, rows):
    """Save a list of dictionaries as a human-readable CSV table."""
    if not rows:
        raise ValueError(f"There are no rows to save in {path.name}")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_scene(scenario):
    """Read hidden truth and make fixed local regions around known sources."""
    query = np.load(MODEL_INPUTS / scenario / "query_positions.npy")
    truth = np.load(EVALUATION_ONLY / scenario / "true_levels.npy")
    sources = np.load(EVALUATION_ONLY / scenario / "source_positions.npy")
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError(f"Invalid query positions in {scenario}")
    if truth.ndim != 2 or truth.shape[1] != len(query):
        raise ValueError(f"Invalid truth shape in {scenario}")
    if sources.ndim != 3 or sources.shape[0] != len(truth) or sources.shape[2] != 2:
        raise ValueError(f"Invalid source positions in {scenario}")
    if not all(np.isfinite(array).all() for array in (query, truth, sources)):
        raise ValueError(f"Non-finite evaluation data in {scenario}")

    # Moving source: these regions follow the source in each frame.
    source_regions = []
    for frame_sources in sources:
        regions = []
        for source_position in frame_sources:
            squared_distance = np.sum((query - source_position)**2, axis=1)
            region = squared_distance <= SOURCE_RADIUS_M**2
            if not np.any(region):
                raise ValueError(f"No grid points near a source in {scenario}")
            regions.append(region)
        source_regions.append(regions)
    return query, truth, sources, source_regions


def grid_shape_and_extent(query):
    """Recover the rectangular map shape and axes from flat query positions."""
    x_values = np.unique(query[:, 0])
    y_values = np.unique(query[:, 1])
    shape = (len(y_values), len(x_values))
    if shape[0] * shape[1] != len(query):
        raise ValueError("Query positions cannot be reshaped into a map")
    extent = (x_values.min(), x_values.max(), y_values.min(), y_values.max())
    return shape, extent


def save_sampling_figure(scenario, query, phone_positions, phone_levels):
    """Show the 100 grid phones coloured by their frame-0 observations."""
    _, extent = grid_shape_and_extent(query)
    fig, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    dots = axis.scatter(
        phone_positions[:, 0], phone_positions[:, 1], c=phone_levels,
        cmap="viridis", s=65, edgecolors="black", linewidths=0.3,
    )
    axis.set(xlim=extent[:2], ylim=extent[2:], aspect="equal",
             xlabel="x (m)", ylabel="y (m)",
             title=f"{LABELS[scenario]}: 100 grid phones, frame 0")
    axis.grid(alpha=0.2)
    fig.colorbar(dots, ax=axis, label="Observed level (dB)")
    fig.savefig(EVALUATION_ONLY / f"sampling_100_grid_{scenario}.png", dpi=160)
    plt.close(fig)


def save_map_comparison(scene_cache, comparison_maps):
    """Put the three scenes' truth, GP prediction and error in nine panels."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for row, scenario in enumerate(SCENARIOS):
        query, truth, _, _ = scene_cache[scenario]
        shape, extent = grid_shape_and_extent(query)
        true_map = truth[0]
        prediction = comparison_maps[scenario]
        error = np.abs(prediction - true_map)
        minimum = min(true_map.min(), prediction.min())
        maximum = max(true_map.max(), prediction.max())
        for column, (values, title) in enumerate((
            (true_map, "True field"),
            (prediction, "GP prediction"),
        )):
            image = axes[row, column].imshow(
                values.reshape(shape), origin="lower", extent=extent,
                vmin=minimum, vmax=maximum,
            )
            axes[row, column].set(title=title, xlabel="x (m)", ylabel="y (m)")
        fig.colorbar(image, ax=axes[row, :2], label="Level (dB)", shrink=0.75)
        error_image = axes[row, 2].imshow(
            error.reshape(shape), origin="lower", extent=extent, vmin=0,
        )
        axes[row, 2].set(title="Absolute error", xlabel="x (m)", ylabel="y (m)")
        fig.colorbar(error_image, ax=axes[row, 2], label="Error (dB)", shrink=0.75)
        axes[row, 0].set_ylabel(f"{LABELS[scenario]}\ny (m)")
    fig.suptitle("GP maps: 100 grid phones, frame 0")
    fig.savefig(EVALUATION_ONLY / "gp_truth_prediction_error_3x3.png", dpi=160)
    plt.close(fig)


def save_three_scene_rmse_lines(overall_rows):
    """Compare the three GP sound fields across phone counts and layouts."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, layout in zip(axes, ("random", "grid", "clustered")):
        for scenario in SCENARIOS:
            group = sorted(
                (row for row in overall_rows
                 if row["scenario"] == scenario and row["layout"] == layout),
                key=lambda row: row["phone_count"],
            )
            axis.plot(
                [row["phone_count"] for row in group],
                [row["RMSE_db"] for row in group], "o-", label=LABELS[scenario],
            )
        axis.set(title=layout.title(), xlabel="Number of phones",
                 ylabel="Whole-map RMSE (dB)")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("GP comparison across three sound fields and sampling layouts")
    fig.savefig(EVALUATION_ONLY / "gp_three_scenario_rmse_lines.png", dpi=160)
    plt.close(fig)


def main():
    manifest_file = PREDICTIONS / "run_manifest.csv"
    if not manifest_file.exists():
        raise FileNotFoundError("Run the sampling and GP model parts first.")
    with manifest_file.open(newline="") as file:
        runs = list(csv.DictReader(file))
    if not runs:
        raise ValueError("The GP run manifest is empty")

    scene_cache = {}
    overall_rows = []
    frame_rows = []
    source_rows = []
    comparison_maps = {}
    comparison_phones = {}
    comparison_readings = {}

    for run in runs:
        scenario = run["scenario"]
        count = int(run["phone_count"])
        layout = run["layout"]
        seed = int(run["seed"])
        if scenario not in scene_cache:
            scene_cache[scenario] = load_scene(scenario)
        query, truth, sources, source_regions = scene_cache[scenario]
        prediction = np.load(PREDICTIONS / run["prediction_file"], mmap_mode="r")
        with np.load(MODEL_INPUTS / run["observations_file"], allow_pickle=False) as sample:
            phone_positions = sample["phone_positions"]
            phone_levels = sample["phone_levels"]
        if prediction.shape != truth.shape or phone_positions.shape != (count, 2):
            raise ValueError(f"Unexpected arrays for {scenario}: {count} {layout}")
        if phone_levels.shape != (len(truth), count):
            raise ValueError(f"Unexpected frame count for {scenario}: {count} {layout}")
        if not np.isfinite(prediction).all():
            raise ValueError(f"Non-finite predictions for {scenario}: {count} {layout}")

        # Prediction minus truth, in dB, for every frame and grid position.
        error = prediction.astype(float) - truth
        absolute_error = np.abs(error)
        local_maes = []
        frame_peak_gaps = []
        for frame in range(len(truth)):
            frame_error = error[frame]
            frame_absolute_error = absolute_error[frame]
            peak_gap = float(prediction[frame].max() - truth[frame].max())
            frame_peak_gaps.append(peak_gap)
            frame_local_maes = []

            for source_id, (source_position, region) in enumerate(
                zip(sources[frame], source_regions[frame]), start=1,
            ):
                nearest_phone = np.linalg.norm(
                    phone_positions - source_position, axis=1,
                ).min()
                local_mae = float(frame_absolute_error[region].mean())
                local_maes.append(local_mae)
                frame_local_maes.append(local_mae)
                source_rows.append({
                    "scenario": scenario,
                    "phone_count": count,
                    "layout": layout,
                    "seed": seed,
                    "frame": frame,
                    "source_id": source_id,
                    "source_x_m": float(source_position[0]),
                    "source_y_m": float(source_position[1]),
                    "nearest_phone_distance_m": float(nearest_phone),
                    "region_radius_m": SOURCE_RADIUS_M,
                    "MAE_near_source_db": local_mae,
                    "local_peak_gap_db": float(
                        prediction[frame, region].max() - truth[frame, region].max()
                    ),
                })

            frame_rows.append({
                "scenario": scenario,
                "phone_count": count,
                "layout": layout,
                "seed": seed,
                "frame": frame,
                "MAE_db": float(frame_absolute_error.mean()),
                "RMSE_db": float(np.sqrt(np.mean(frame_error**2))),
                "Bias_db": float(frame_error.mean()),
                "P95_absolute_error_db": float(np.percentile(frame_absolute_error, 95)),
                "global_peak_gap_db": peak_gap,
                "mean_source_MAE_db": float(np.mean(frame_local_maes)),
            })

        overall = {
            "scenario": scenario,
            "phone_count": count,
            "layout": layout,
            "seed": seed,
            "frames": len(truth),
            "MAE_db": float(absolute_error.mean()),
            "RMSE_db": float(np.sqrt(np.mean(error**2))),
            "Bias_db": float(error.mean()),
            "P95_absolute_error_db": float(np.percentile(absolute_error, 95)),
            "mean_global_peak_gap_db": float(np.mean(frame_peak_gaps)),
            "mean_source_MAE_db": float(np.mean(local_maes)),
            "worst_source_MAE_db": float(np.max(local_maes)),
        }
        overall_rows.append(overall)
        print(f"{scenario:16s} {count:3d} {layout:9s}: "
              f"RMSE {overall['RMSE_db']:.2f} dB, "
              f"worst source MAE {overall['worst_source_MAE_db']:.2f} dB")
        if (count, layout) == (100, "grid"):
            comparison_maps[scenario] = np.array(prediction[0], copy=True)
            comparison_phones[scenario] = phone_positions.copy()
            comparison_readings[scenario] = phone_levels[0].copy()

    write_rows(EVALUATION_ONLY / "overall_metrics.csv", overall_rows)
    write_rows(EVALUATION_ONLY / "frame_metrics.csv", frame_rows)
    write_rows(EVALUATION_ONLY / "per_source_metrics.csv", source_rows)

    # Make one compact CSV so the same sampling choice can be compared across
    # all three scenes. Different scenes have different difficulty, of course.
    by_key = {(row["scenario"], row["phone_count"], row["layout"]): row
              for row in overall_rows}
    comparison_rows = []
    for count, layout in sorted({(row["phone_count"], row["layout"])
                                 for row in overall_rows}):
        if not all((scenario, count, layout) in by_key for scenario in SCENARIOS):
            raise ValueError(f"Missing scenario for {count} {layout}")
        comparison_row = {"phone_count": count, "layout": layout}
        for scenario in SCENARIOS:
            row = by_key[(scenario, count, layout)]
            comparison_row[f"{scenario}_RMSE_db"] = row["RMSE_db"]
            comparison_row[f"{scenario}_worst_source_MAE_db"] = row["worst_source_MAE_db"]
        comparison_rows.append(comparison_row)
    write_rows(EVALUATION_ONLY / "three_scenario_comparison.csv", comparison_rows)

    for scenario in SCENARIOS:
        if scenario not in comparison_maps:
            raise ValueError(f"Missing 100-grid example for {scenario}")
        query, _, _, _ = scene_cache[scenario]
        save_sampling_figure(
            scenario, query,
            comparison_phones[scenario], comparison_readings[scenario],
        )
    save_map_comparison(scene_cache, comparison_maps)
    save_three_scene_rmse_lines(overall_rows)
    print("\nScores and figures saved in:", EVALUATION_ONLY)


if __name__ == "__main__":
    main()
