"""Prepare one single-source dataset using Andrei's code. No GP runs here."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # Save the figure instead of opening a plot window.
import matplotlib.pyplot as plt


# Find Andrei's files in the folder next to austin/.
folder = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(folder / "andreib"))
import point_source
from extract_sound import prepare_interpolation_data


# 1. Use Andrei's sound field to make 20 random phone measurements.
all_data = prepare_interpolation_data(
    point_source.sound_maps,
    point_source.x,
    point_source.y,
    phone_counts=[20],
    layouts=["random"],
    seed=42,
)
data = all_data[(20, "random")]

# 2. Give the four arrays short names.
phone_positions = data["phone_positions"]  # Where the 20 phones are.
phone_levels = data["phone_levels"]        # What the 20 phones measure.
query_positions = data["query_positions"]  # Where we will predict later.
true_levels = data["true_levels"]          # The answer, used only for scoring.

print("Phone positions:", phone_positions.shape)  # (20, 2)
print("Phone levels:", phone_levels.shape)        # (1, 20)
print("Query positions:", query_positions.shape)  # (40401, 2)
print("True levels:", true_levels.shape)          # (1, 40401)

# 3. Draw the complete simulated field and the 20 observed points.
true_map = true_levels[0].reshape(201, 201)
lowest = true_levels.min()
highest = true_levels.max()
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

image = axes[0].imshow(
    true_map, origin="lower", extent=(0, 100, 0, 100),
    vmin=lowest, vmax=highest,
)
axes[0].scatter(phone_positions[:, 0], phone_positions[:, 1], color="white")
axes[0].set_title("Full sound field and phone positions")

axes[1].scatter(
    phone_positions[:, 0], phone_positions[:, 1],
    c=phone_levels[0], vmin=lowest, vmax=highest, s=70,
)
axes[1].set_title("Only the 20 phone readings")

for axis in axes:
    axis.set(xlim=(0, 100), ylim=(0, 100), xlabel="x (m)", ylabel="y (m)")
    axis.set_aspect("equal")

fig.colorbar(image, ax=axes, label="Level (dB)")
output = Path(__file__).with_name("gp_single_source_inputs.png")
fig.savefig(output, dpi=160)
plt.close("all")
print("Saved figure:", output)
