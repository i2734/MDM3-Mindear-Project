"""GP single-source preparation: inspect one dataset from Andrei's simulator.

This is the first data-preparation step for a future GP model. It does not fit
or run a GP. It uses Andrei's existing single-source scene and sampling helper
to obtain one reproducible set of 20 simulated phone measurements, checks the
array shapes and coordinate order, and plots the full truth beside the sparse
observations.
"""

from pathlib import Path
import os
import sys
import tempfile

# Put Matplotlib's cache in a writable temporary directory before importing it.
plot_cache = Path(tempfile.gettempdir()) / "mdm3_step1_matplotlib"
plot_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))

import matplotlib

# Andrei's scene module creates a figure when imported. Agg avoids opening a
# graphical window and lets this script save its figure reliably.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# This file is in austin/; the sibling andreib/ directory contains the source
# scene and the function that samples phone measurements from it.
SIMULATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATION_ROOT / "andreib"))
import point_source  # noqa: E402
from extract_sound import prepare_interpolation_data  # noqa: E402


def main() -> None:
    # Start with one static source and one reproducible sampling configuration.
    # The seed ensures that repeated runs generate the same phone positions.
    n_phones = 20
    layout = "random"
    seed = 42

    # sound_maps has axes (time, y, x). The source scene has one time frame and
    # a 201-by-201 grid covering 0 to 100 metres at 0.5-metre intervals.
    prepared = prepare_interpolation_data(
        sound_maps=point_source.sound_maps,
        x=point_source.x,
        y=point_source.y,
        phone_counts=[n_phones],
        layouts=[layout],
        seed=seed,
    )

    # The dictionary key is (number of phones, spatial layout). These are the
    # four arrays associated with the single configuration chosen above.
    sample = prepared[(n_phones, layout)]
    phone_positions = sample["phone_positions"]  # (N, 2): each phone's x and y
    phone_levels = sample["phone_levels"]        # (T, N): observed levels
    query_positions = sample["query_positions"]  # (M, 2): prediction locations
    true_levels = sample["true_levels"]          # (T, M): scoring-only answers

    # Check the shared data contract before any interpolation is attempted.
    # If the scene or grid changes later, these checks prompt us to inspect it.
    grid_shape = (len(point_source.y), len(point_source.x))
    n_queries = grid_shape[0] * grid_shape[1]
    assert point_source.sound_maps.shape == (1, *grid_shape)
    assert phone_positions.shape == (n_phones, 2)
    assert phone_levels.shape == (1, n_phones)
    assert query_positions.shape == (n_queries, 2)
    assert true_levels.shape == (1, n_queries)
    assert all(np.isfinite(array).all() for array in sample.values())

    # Andrei flattens a meshgrid row by row: x changes first, then y. The same
    # flattened index must refer to the same location in query_positions and
    # true_levels. The truth is checked here, but a future GP must not read it.
    grid_x, grid_y = np.meshgrid(point_source.x, point_source.y)
    expected_queries = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    np.testing.assert_array_equal(query_positions, expected_queries)
    np.testing.assert_array_equal(true_levels[0], point_source.sound_maps[0].ravel())

    print("Scene: one stationary point source")
    print(f"Layout: {layout}; phones: {n_phones}; random seed: {seed}")
    print("Sound-map shape:", point_source.sound_maps.shape)
    for name in ("phone_positions", "phone_levels", "query_positions", "true_levels"):
        print(f"{name:>15} shape: {sample[name].shape}")
    print("First three query positions (m):", query_positions[:3].tolist())
    print("Query position 202 (m):", query_positions[201].tolist())
    print(f"Truth range: {true_levels.min():.3f} to {true_levels.max():.3f} dB")
    print(f"Observed range: {phone_levels.min():.3f} to {phone_levels.max():.3f} dB")

    # Andrei generates continuous phone coordinates, then reads the value at
    # the nearest grid cell. Print both coordinates to expose this small offset.
    print("First five simulated phone measurements:")
    print("Phone | continuous position (m) | nearest grid cell (m) | level (dB)")
    for i, (px, py) in enumerate(phone_positions[:5]):
        x_index = int(np.argmin(np.abs(point_source.x - px)))
        y_index = int(np.argmin(np.abs(point_source.y - py)))
        nearest_x = point_source.x[x_index]
        nearest_y = point_source.y[y_index]
        observed = phone_levels[0, i]
        np.testing.assert_equal(observed, point_source.sound_maps[0, y_index, x_index])
        print(f"{i + 1:>5} | ({px:6.2f}, {py:6.2f}) "
              f"| ({nearest_x:6.2f}, {nearest_y:6.2f}) | {observed:8.3f}")

    # The left panel shows the answer known to the simulator. The right panel
    # shows only the 20 values that a future spatial predictor may receive.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    level_min, level_max = float(true_levels.min()), float(true_levels.max())
    truth_image = axes[0].imshow(
        true_levels[0].reshape(grid_shape),
        origin="lower", extent=(0, 100, 0, 100),
        vmin=level_min, vmax=level_max, cmap="viridis",
    )
    axes[0].scatter(phone_positions[:, 0], phone_positions[:, 1],
                    s=22, facecolors="white", edgecolors="black")
    axes[0].set_title("Simulated truth + 20 sample positions")
    axes[1].scatter(phone_positions[:, 0], phone_positions[:, 1],
                    c=phone_levels[0], s=70, edgecolors="black",
                    vmin=level_min, vmax=level_max, cmap="viridis")
    axes[1].set_title("Only the 20 observed levels")
    for axis in axes:
        axis.set(xlim=(0, 100), ylim=(0, 100), xlabel="x (m)", ylabel="y (m)")
        axis.set_aspect("equal")
    fig.colorbar(truth_image, ax=axes, label="Simulated level (dB)", shrink=0.8)

    output = Path(__file__).with_name("gp_single_source_inputs.png")
    fig.savefig(output, dpi=160)
    plt.close("all")
    print("Figure saved to:", output)
    print("STEP1_DATA_CHECKS_PASSED")


if __name__ == "__main__":
    main()
