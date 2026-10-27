import numpy as np


def generate_phone_positions(x, y, n_phones, layout, seed=42):
    """Generate the coordinates where phones record sound."""

    rng = np.random.default_rng(seed)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    if layout == "random":
        phone_positions = np.column_stack((
            rng.uniform(x_min, x_max, n_phones),
            rng.uniform(y_min, y_max, n_phones)
        ))

    elif layout == "grid":
        n_cols = int(np.ceil(np.sqrt(n_phones)))
        n_rows = int(np.ceil(n_phones / n_cols))

        positions = []

        for row in range(n_rows):
            for col in range(n_cols):
                if len(positions) == n_phones:
                    break

                px = x_min + (col + 0.5) * (x_max - x_min) / n_cols
                py = y_min + (row + 0.5) * (y_max - y_min) / n_rows

                positions.append((px, py))

        phone_positions = np.array(positions)

    elif layout == "clustered":
        centre_x = rng.uniform(x_min, x_max)
        centre_y = rng.uniform(y_min, y_max)

        phone_positions = np.column_stack((
            np.clip(
                rng.normal(centre_x, (x_max - x_min) / 10, n_phones),
                x_min, x_max
            ),
            np.clip(
                rng.normal(centre_y, (y_max - y_min) / 10, n_phones),
                y_min, y_max
            )
        ))

    else:
        raise ValueError(
            "layout must be 'grid', 'random' or 'clustered'"
        )

    return phone_positions


def extract_measurements(
    sound_maps, x, y, phone_positions
):
    """Extract sound levels at phone positions for every time step."""

    sound_maps = np.asarray(sound_maps)

    if sound_maps.ndim == 2:
        sound_maps = sound_maps[np.newaxis, :, :]

    T = sound_maps.shape[0]
    N = len(phone_positions)

    # Each row is a time step; each column is a recording phone.
    phone_levels = np.zeros((T, N))

    for i, (px, py) in enumerate(phone_positions):
        x_idx = np.argmin(np.abs(x - px))
        y_idx = np.argmin(np.abs(y - py))

        phone_levels[:, i] = sound_maps[:, y_idx, x_idx]

    return phone_levels


def prepare_interpolation_data(
    sound_maps, x, y, phone_counts, layouts, seed=42
):
    """Prepare phone measurements and reference values for interpolation."""

    sound_maps = np.asarray(sound_maps)

    if sound_maps.ndim == 2:
        sound_maps = sound_maps[np.newaxis, :, :]

    # Locations where the interpolation models will predict sound levels.
    X, Y = np.meshgrid(x, y)
    query_positions = np.column_stack((X.ravel(), Y.ravel()))

    # True sound levels at every query position and time step.
    true_levels = sound_maps.reshape(sound_maps.shape[0], -1)

    prepared_data = {}

    for layout in layouts:
        for n_phones in phone_counts:

            # Locations where phones record sound.
            phone_positions = generate_phone_positions(
                x, y, n_phones, layout, seed
            )

            # Simulated measurements recorded at those locations.
            phone_levels = extract_measurements(
                sound_maps, x, y, phone_positions
            )

            prepared_data[(n_phones, layout)] = {
                "phone_positions": phone_positions,
                "phone_levels": phone_levels,
                "query_positions": query_positions,
                "true_levels": true_levels
            }

    return prepared_data