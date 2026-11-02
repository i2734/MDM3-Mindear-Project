"""Step 2: predict the single-source sound map from 20 simulated phone readings."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # Save a picture without opening a plot window.
import matplotlib.pyplot as plt
import numpy as np


# Andrei's files are in the folder next to austin/.
folder = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(folder / "andreib"))
import point_source
from extract_sound import prepare_interpolation_data


def rbf_kernel(first, second):
    """Give nearby positions a stronger connection than distant positions."""
    length_scale = 15.0  # metres; a starting choice, not a fitted value
    signal_std = 15.0    # dB; a starting choice, not a measured value

    # The result has one row per position in first and one column per position
    # in second. Each entry depends on the squared distance between two points.
    difference = first[:, None, :] - second[None, :, :]
    distance_squared = np.sum(difference**2, axis=2)
    return signal_std**2 * np.exp(
        -distance_squared / (2 * length_scale**2)
    )


def predict_gp(phone_positions, phone_levels, query_positions):
    """Return one predicted sound map per time frame, with shape (T, M)."""
    # K describes how strongly the phone positions are related to each other.
    k_phones = rbf_kernel(phone_positions, phone_positions)
    k_phones += 1e-6 * np.eye(len(phone_positions))  # Numerical stability.

    # K_star describes how each map position is related to the phone positions.
    k_query_phones = rbf_kernel(query_positions, phone_positions)

    maps = []
    for readings in phone_levels:  # The simple example has just one time frame.
        average = readings.mean()
        change_from_average = readings - average
        weights = np.linalg.solve(k_phones, change_from_average)
        predicted_map = average + k_query_phones @ weights
        maps.append(predicted_map)

    return np.array(maps)


# Use exactly the same scene, phone count, layout and seed as Step 1.
data = prepare_interpolation_data(
    point_source.sound_maps,
    point_source.x,
    point_source.y,
    phone_counts=[20],
    layouts=["random"],
    seed=42,
)[(20, "random")]

# Only these three arrays enter the GP. The true map stays out of prediction.
phone_positions = data["phone_positions"]
phone_levels = data["phone_levels"]
query_positions = data["query_positions"]

prediction = predict_gp(phone_positions, phone_levels, query_positions)
print("Prediction shape:", prediction.shape)  # (1, 40401)
print(f"Predicted range: {prediction.min():.3f} to {prediction.max():.3f} dB")

# Save the prediction as a table: x coordinate, y coordinate, predicted level.
output_csv = Path(__file__).with_name("gp_single_source_predictions.csv")
table = np.column_stack((query_positions, prediction[0]))
np.savetxt(
    output_csv, table, delimiter=",",
    header="x_m,y_m,predicted_level_db", comments="", fmt="%.6f",
)

# Draw the GP prediction. White dots mark the 20 readings used by the model.
fig, axis = plt.subplots(figsize=(5, 4))
image = axis.imshow(
    prediction[0].reshape(201, 201),
    origin="lower", extent=(0, 100, 0, 100),
)
axis.scatter(phone_positions[:, 0], phone_positions[:, 1], color="white")
axis.set(title="GP prediction from 20 phone readings",
         xlabel="x (m)", ylabel="y (m)")
fig.colorbar(image, ax=axis, label="Predicted level (dB)")
fig.tight_layout()
output_png = Path(__file__).with_name("gp_single_source_prediction.png")
fig.savefig(output_png, dpi=160)
plt.close("all")

print("Saved table:", output_csv)
print("Saved figure:", output_png)
