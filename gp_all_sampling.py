"""Part 1: make the same sampling experiment for all three Andrei scenes.

Run this file before gp_all_sources_model.py. The model_inputs directory has
only observed phone values and prediction locations. Complete true maps and
source locations stay in evaluation_only and are used only by Part 3.

Arrays use one convention throughout:
    phone_positions: (N, 2)       fixed phones for all frames
    phone_levels:    (T, N)       one row per frame
    query_positions: (M, 2)       the same 201 x 201 prediction grid
    true_levels:     (T, M)       hidden from the GP model
"""

import csv
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Andrei's moving-source file creates an animation for display. This pipeline
# reads its sound maps without rendering that animation, so its exit warning
# would be misleading here.
warnings.filterwarnings(
    "ignore", message="Animation was deleted without rendering anything.*",
    category=UserWarning, module="matplotlib.animation",
)


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "andreib"))
import point_source
import multiple_point_sources
import single_moving_point_source
from extract_sound import prepare_interpolation_data


OUTPUT = HERE / "gp_all_sources_output"
MODEL_INPUTS = OUTPUT / "model_inputs"
EVALUATION_ONLY = OUTPUT / "evaluation_only"
PHONE_COUNTS = (5, 10, 20, 50, 100)
LAYOUTS = ("random", "grid", "clustered")
SEED = 42

# These names are used consistently in all three parts and in output tables.
SCENES = (
    ("single_static", point_source),
    ("multiple_static", multiple_point_sources),
    ("single_moving", single_moving_point_source),
)


def known_source_positions(name, scene):
    """Return (T, number_of_sources, 2), for evaluation only."""
    if name == "single_static":
        return np.array([[[scene.source_x_positions[0], scene.source_y[0]]]])
    if name == "multiple_static":
        return np.array([[source[:2] for source in scene.sources]], dtype=float)
    # Andrei's moving source follows one (x, y) path over 101 frames.
    return np.column_stack((scene.source_x_positions, scene.source_y))[:, None, :]


def main():
    MODEL_INPUTS.mkdir(parents=True, exist_ok=True)
    EVALUATION_ONLY.mkdir(parents=True, exist_ok=True)

    with (OUTPUT / "sampling_manifest.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("scenario", "phone_count", "layout", "seed", "observations_file"))

        for name, scene in SCENES:
            model_folder = MODEL_INPUTS / name
            truth_folder = EVALUATION_ONLY / name
            model_folder.mkdir(exist_ok=True)
            truth_folder.mkdir(exist_ok=True)

            # This is Andrei's original sampler; no new sampling algorithm.
            data = prepare_interpolation_data(
                scene.sound_maps, scene.x, scene.y,
                phone_counts=PHONE_COUNTS, layouts=LAYOUTS, seed=SEED,
            )
            first = data[(PHONE_COUNTS[0], LAYOUTS[0])]
            query_positions = first["query_positions"]
            true_levels = first["true_levels"]
            source_positions = known_source_positions(name, scene)
            frames = len(true_levels)
            if query_positions.shape != (201 * 201, 2):
                raise ValueError(f"Unexpected grid shape in {name}")
            if true_levels.shape != (frames, len(query_positions)):
                raise ValueError(f"Unexpected true-level shape in {name}")
            if source_positions.shape[0] != frames or source_positions.shape[2] != 2:
                raise ValueError(f"Unexpected source-position shape in {name}")

            np.save(model_folder / "query_positions.npy", query_positions)
            np.save(truth_folder / "true_levels.npy", true_levels)
            np.save(truth_folder / "source_positions.npy", source_positions)
            print(f"\n{name}: {frames} frame(s), {len(query_positions)} grid positions")

            for layout in LAYOUTS:
                for count in PHONE_COUNTS:
                    sample = data[(count, layout)]
                    phone_positions = sample["phone_positions"]
                    phone_levels = sample["phone_levels"]
                    if phone_positions.shape != (count, 2) or phone_levels.shape != (frames, count):
                        raise ValueError(f"Unexpected phone arrays in {name}: {count} {layout}")
                    if not np.array_equal(sample["query_positions"], query_positions):
                        raise ValueError(f"Query grid changed in {name}: {count} {layout}")

                    filename = f"observations_{count}_{layout}.npz"
                    relative_file = Path(name) / filename
                    np.savez_compressed(
                        model_folder / filename,
                        phone_positions=phone_positions,
                        phone_levels=phone_levels,
                    )
                    writer.writerow((name, count, layout, SEED, relative_file.as_posix()))
                    print(f"  {count:3d} {layout:9s}: positions {phone_positions.shape}, "
                          f"levels {phone_levels.shape}")

    # Andrei's source modules make example figures when imported. Close them;
    # this sampling step does not need to display or save those figures.
    plt.close("all")
    print("\nModel inputs:", MODEL_INPUTS)
    print("Hidden truth for evaluation:", EVALUATION_ONLY)


if __name__ == "__main__":
    main()
