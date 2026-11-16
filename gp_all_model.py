"""Part 2: run one RBF Gaussian Process model for every saved configuration.

Run gp_all_sources_sampling.py first. This file reads only model_inputs, so it
does not use the complete true field or known source positions to predict.
For a moving source, it predicts each frame independently with the same GP.
This is a sequence of spatial GPs, not a space-time GP.
"""

import csv
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "gp_all_sources_output"
MODEL_INPUTS = OUTPUT / "model_inputs"
PREDICTIONS = OUTPUT / "gp_predictions"

# Keep the settings equal to the existing single- and multiple-source GP tests.
LENGTH_SCALE_M = 15.0
SIGNAL_STD_DB = 15.0
NOISE_STD_DB = 0.5  # Assumed model noise; not measured phone-calibration noise.
JITTER = 1e-6      # Small numerical aid, separate from the model noise.


def rbf_kernel(first, second):
    """Covariance between every position in first and every position in second."""
    first_size = np.sum(first**2, axis=1)[:, None]
    second_size = np.sum(second**2, axis=1)[None, :]
    distance_squared = first_size + second_size - 2 * first @ second.T
    distance_squared = np.maximum(distance_squared, 0.0)
    return SIGNAL_STD_DB**2 * np.exp(
        -distance_squared / (2 * LENGTH_SCALE_M**2)
    )


def predict_gp(phone_positions, phone_levels, query_positions):
    """Take (N,2), (T,N), (M,2) and return predicted maps (T,M)."""
    n_phones = len(phone_positions)
    if phone_positions.shape != (n_phones, 2):
        raise ValueError("phone_positions must have shape (N, 2)")
    if phone_levels.ndim != 2 or phone_levels.shape[1] != n_phones:
        raise ValueError("phone_levels must have shape (T, N)")
    if query_positions.ndim != 2 or query_positions.shape[1] != 2:
        raise ValueError("query_positions must have shape (M, 2)")

    # K describes relationships between observed phone locations.
    k_phones = rbf_kernel(phone_positions, phone_positions)
    k_phones += (NOISE_STD_DB**2 + JITTER) * np.eye(n_phones)
    lower = np.linalg.cholesky(k_phones)

    # K_star describes relationships between prediction and phone locations.
    k_query_phones = rbf_kernel(query_positions, phone_positions)
    means = phone_levels.mean(axis=1, keepdims=True)  # (T, 1)
    centred = phone_levels - means                    # (T, N)

    # Solve for all frames at once. Cholesky avoids forming a matrix inverse.
    weights = np.linalg.solve(lower.T, np.linalg.solve(lower, centred.T))
    return (k_query_phones @ weights).T + means       # (T, M)


def main():
    manifest_file = OUTPUT / "sampling_manifest.csv"
    if not manifest_file.exists():
        raise FileNotFoundError("Run gp_all_sources_sampling.py first.")
    with manifest_file.open(newline="") as file:
        configurations = list(csv.DictReader(file))
    if not configurations:
        raise ValueError("sampling_manifest.csv contains no configurations")
    PREDICTIONS.mkdir(exist_ok=True)

    runs = []
    for config in configurations:
        scenario = config["scenario"]
        count = int(config["phone_count"])
        layout = config["layout"]
        seed = int(config["seed"])
        query_positions = np.load(MODEL_INPUTS / scenario / "query_positions.npy")
        with np.load(MODEL_INPUTS / config["observations_file"], allow_pickle=False) as sample:
            phone_positions = sample["phone_positions"]
            phone_levels = sample["phone_levels"]
        if phone_positions.shape != (count, 2) or phone_levels.shape[1] != count:
            raise ValueError(f"Invalid input shapes for {scenario}: {count} {layout}")
        if not all(np.isfinite(array).all() for array in
                   (phone_positions, phone_levels, query_positions)):
            raise ValueError(f"Non-finite input for {scenario}: {count} {layout}")

        prediction = predict_gp(phone_positions, phone_levels, query_positions)
        if prediction.shape != (len(phone_levels), len(query_positions)):
            raise ValueError(f"Invalid output shape for {scenario}: {count} {layout}")
        if not np.isfinite(prediction).all():
            raise ValueError(f"Non-finite prediction for {scenario}: {count} {layout}")

        prediction_folder = PREDICTIONS / scenario
        prediction_folder.mkdir(exist_ok=True)
        prediction_file = Path(scenario) / f"prediction_{count}_{layout}.npy"
        np.save(PREDICTIONS / prediction_file, prediction.astype(np.float32))
        runs.append((scenario, count, layout, seed, len(phone_levels),
                     len(query_positions), config["observations_file"],
                     prediction_file.as_posix()))
        print(f"{scenario:16s} {count:3d} {layout:9s}: "
              f"prediction {prediction.shape}, "
              f"{prediction.min():.2f} to {prediction.max():.2f} dB")

    with (PREDICTIONS / "run_manifest.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("scenario", "phone_count", "layout", "seed", "frames",
                         "grid_points", "observations_file", "prediction_file"))
        writer.writerows(runs)
    with (PREDICTIONS / "model_settings.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("kernel", "length_scale_m", "signal_std_db",
                         "assumed_noise_std_db", "jitter", "frame_model", "saved_dtype"))
        writer.writerow(("RBF", LENGTH_SCALE_M, SIGNAL_STD_DB, NOISE_STD_DB,
                         JITTER, "independent_spatial_GP", "float32"))
    print("\nSaved predictions in:", PREDICTIONS)


if __name__ == "__main__":
    main()
