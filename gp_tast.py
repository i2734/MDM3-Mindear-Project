"""Step 3: compare the saved GP map with Andrei's full single-source map."""

import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # Save the figure without opening a window.
import matplotlib.pyplot as plt
import numpy as np


# Use the same Andrei scene and data-preparation function as Step 2.
folder = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(folder / "andreib"))
import point_source
from extract_sound import prepare_interpolation_data


# Read the prediction saved by Step 2. This file has x, y and predicted dB.
prediction_csv = Path(__file__).with_name("gp_single_source_predictions.csv")
if not prediction_csv.exists():
    raise FileNotFoundError("Run gp_single_source_step2.py before Step 3.")
prediction_table = np.loadtxt(prediction_csv, delimiter=",", skiprows=1)

# Get the hidden truth only now, after the GP prediction has been saved.
data = prepare_interpolation_data(
    point_source.sound_maps,
    point_source.x,
    point_source.y,
    phone_counts=[20],
    layouts=["random"],
    seed=42,
)[(20, "random")]
query_positions = data["query_positions"]
true_levels = data["true_levels"][0]  # One time frame, 40,401 grid points.

# Check that each CSV prediction is paired with the correct truth position.
if prediction_table.shape != (len(true_levels), 3):
    raise ValueError("The prediction CSV does not match the Step 2 grid size.")
if not np.allclose(prediction_table[:, :2], query_positions, atol=1e-6, rtol=0):
    raise ValueError("The prediction CSV coordinates do not match Andrei's grid.")
predicted_levels = prediction_table[:, 2]
if not np.all(np.isfinite(predicted_levels)):
    raise ValueError("The prediction CSV contains invalid sound levels.")

# A positive error means the GP predicted too high; negative means too low.
error = predicted_levels - true_levels
absolute_error = np.abs(error)
mae = np.mean(absolute_error)
rmse = np.sqrt(np.mean(error**2))
bias = np.mean(error)
p95 = np.percentile(absolute_error, 95)

# 80 dB is an example analysis threshold for this scene, not a safety limit.
high_threshold = 80.0
high_points = true_levels >= high_threshold
high_mae = np.mean(absolute_error[high_points])

# Compare the two peak heights. Andrei's source has several equally high
# centre cells, so picking just the first maximum cell would be arbitrary.
peak_gap = np.max(predicted_levels) - np.max(true_levels)

metrics = [
    ("MAE", mae, "dB"),
    ("RMSE", rmse, "dB"),
    ("Bias (predicted - true)", bias, "dB"),
    ("P95 absolute error", p95, "dB"),
    ("MAE where true level >= 80 dB", high_mae, "dB"),
    ("Peak level gap (predicted max - true max)", peak_gap, "dB"),
]

# Save the numbers so later methods can use the same scoring columns.
metrics_csv = Path(__file__).with_name("gp_single_source_step3_metrics.csv")
with metrics_csv.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["metric", "value", "unit"])
    for name, value, unit in metrics:
        writer.writerow([name, f"{value:.6f}", unit])

print(f"Compared {len(true_levels)} grid points.")
for name, value, unit in metrics:
    print(f"{name}: {value:.3f} {unit}")
print(f"High-level points (true >= 80 dB): {np.sum(high_points)}")

# Use one colour scale for truth and prediction so their colours are comparable.
map_shape = point_source.sound_maps.shape[1:]  # (201, 201)
colour_min = min(true_levels.min(), predicted_levels.min())
colour_max = max(true_levels.max(), predicted_levels.max())
extent = (point_source.x.min(), point_source.x.max(),
          point_source.y.min(), point_source.y.max())
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

for axis, levels, title in zip(
    axes[:2], (true_levels, predicted_levels), ("Andrei's truth", "GP prediction")
):
    image = axis.imshow(levels.reshape(map_shape), origin="lower", extent=extent,
                        vmin=colour_min, vmax=colour_max)
    axis.set(title=title, xlabel="x (m)", ylabel="y (m)")
fig.colorbar(image, ax=axes[:2], label="Sound level (dB)", shrink=0.85)

error_image = axes[2].imshow(absolute_error.reshape(map_shape), origin="lower",
                             extent=extent, vmin=0)
axes[2].set(title="Absolute error", xlabel="x (m)", ylabel="y (m)")
fig.colorbar(error_image, ax=axes[2], label="Error (dB)", shrink=0.85)

figure_png = Path(__file__).with_name("gp_single_source_step3_comparison.png")
fig.savefig(figure_png, dpi=160)
plt.close("all")

print("Saved metrics:", metrics_csv)
print("Saved figure:", figure_png)
