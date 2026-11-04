"""Venue-independent phone placement and extraction for regular 2D sound grids.

allowed_mask[y_index, x_index] specifies audience-accessible grid cells.
All generated phones lie exactly on permitted simulation grid points.
"""

import numpy as np


def _validate(x, y, allowed_mask):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if (
        x.ndim != 1
        or y.ndim != 1
        or not np.all(np.diff(x) > 0)
        or not np.all(np.diff(y) > 0)
    ):
        raise ValueError("x and y must be strictly increasing 1D arrays")

    mask = (
        np.ones((len(y), len(x)), bool)
        if allowed_mask is None
        else np.asarray(allowed_mask, dtype=bool)
    )

    if mask.shape != (len(y), len(x)) or not mask.any():
        raise ValueError(
            "allowed_mask must have shape (len(y), len(x)) "
            "and contain True cells"
        )

    X, Y = np.meshgrid(x, y)

    candidates = np.column_stack((
        X[mask],
        Y[mask]
    ))

    return x, y, mask, candidates


def _spread_select(points, n):
    """Greedy max-min subset of rectangular lattice nodes."""

    chosen = np.empty(n, dtype=int)

    chosen[0] = np.argmin(
        np.sum((points - points.mean(axis=0))**2, axis=1)
    )

    nearest = np.sum(
        (points - points[chosen[0]])**2,
        axis=1
    )

    nearest[chosen[0]] = -1

    for i in range(1, n):
        chosen[i] = np.argmax(nearest)

        new = np.sum(
            (points - points[chosen[i]])**2,
            axis=1
        )

        nearest = np.minimum(nearest, new)
        nearest[chosen[:i + 1]] = -1

    return points[chosen]


def generate_phone_positions(
    x, y, n_phones, layout, seed=42, allowed_mask=None
):
    """
    Generate microphone positions inside the permitted region.

    Layouts:
        random
        grid
        clustered
    """

    x, y, mask, candidates = _validate(
        x, y, allowed_mask
    )

    if (
        not isinstance(n_phones, (int, np.integer))
        or not 1 <= n_phones <= len(candidates)
    ):
        raise ValueError(
            "n_phones must be between 1 and the number of permitted cells"
        )

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # RANDOM
    # --------------------------------------------------------

    if layout == "random":

        indices = rng.choice(
            len(candidates),
            n_phones,
            replace=False
        )

        return candidates[indices]

    # --------------------------------------------------------
    # CLUSTERED
    # --------------------------------------------------------

    if layout == "clustered":

        centre = candidates[
            rng.integers(len(candidates))
        ]

        scales = np.maximum(
            [np.ptp(x) / 10, np.ptp(y) / 10],
            1e-12
        )

        score = (
            -0.5 * np.sum(
                ((candidates - centre) / scales)**2,
                axis=1
            )
            + rng.gumbel(size=len(candidates))
        )

        indices = np.argpartition(
            score, -n_phones
        )[-n_phones:]

        return candidates[indices]

    # --------------------------------------------------------
    # RECTANGULAR GRID
    # --------------------------------------------------------

    
    if layout == "grid":
        # Global rectangular grid across the FULL map.
        # Invalid points are removed, not repositioned.

        xmin, xmax = x[0], x[-1]
        ymin, ymax = y[0], y[-1]

        best = None
        best_score = float("inf")

        # Search grid resolutions until enough valid
        # points are found inside the audience mask.
        max_side = min(
            max(len(x), len(y)),
            max(
                20,
                int(3 * np.sqrt(n_phones / mask.mean())) + 8
            )
        )

        minimum = 2 if n_phones >= 4 else 1

        for rows in range(minimum, max_side + 1):
            for cols in range(minimum, max_side + 1):

                if rows * cols < n_phones:
                    continue

                # Construct the full rectangular lattice
                gx = np.linspace(
                    xmin, xmax, cols + 2
                )[1:-1]

                gy = np.linspace(
                    ymin, ymax, rows + 2
                )[1:-1]

                xx, yy = np.meshgrid(gx, gy)

                # Snap lattice nodes to simulation grid
                ix = np.abs(
                    x[None, :] - xx.ravel()[:, None]
                ).argmin(axis=1)

                iy = np.abs(
                    y[None, :] - yy.ravel()[:, None]
                ).argmin(axis=1)

                # Remove nodes outside the audience region
                keep = mask[iy, ix]

                flat = np.unique(
                    iy[keep] * len(x) + ix[keep]
                )

                if len(flat) < n_phones:
                    continue

                points = np.column_stack((
                    x[flat % len(x)],
                    y[flat // len(x)]
                ))

                # Prefer grids with few surplus points
                # and similar horizontal/vertical spacing
                dx = (xmax - xmin) / (cols + 1)
                dy = (ymax - ymin) / (rows + 1)

                spacing_penalty = abs(
                    np.log(
                        max(dx, 1e-12) /
                        max(dy, 1e-12)
                    )
                )

                score = (
                    len(points) - n_phones
                    + 0.35 * spacing_penalty
                )

                if score < best_score:
                    best_score = score
                    best = points

        if best is None:
            raise RuntimeError(
                "Cannot place requested number of grid phones"
            )

        if len(best) == n_phones:
            return best

        # Remove surplus nodes without moving the others.
        # All remaining nodes stay on the original lattice.
        return _spread_select(best, n_phones)


    raise ValueError(
        "layout must be 'random', 'grid', or 'clustered'"
    )


def extract_measurements(
    sound_maps, x, y, phone_positions, allowed_mask=None
):
    """Extract simulated measurements at microphone positions."""

    x, y, mask, _ = _validate(
        x, y, allowed_mask
    )

    maps = np.asarray(sound_maps, dtype=float)

    if maps.ndim == 2:
        maps = maps[None, :, :]

    if maps.ndim != 3 or maps.shape[1:] != mask.shape:
        raise ValueError(
            "sound_maps must have shape (y, x) or (time, y, x)"
        )

    pos = np.asarray(phone_positions, dtype=float)

    if (
        pos.ndim != 2
        or pos.shape[1] != 2
        or not np.isfinite(pos).all()
    ):
        raise ValueError(
            "phone_positions must be a finite array of shape (N, 2)"
        )

    if np.any(
        (pos[:, 0] < x[0])
        | (pos[:, 0] > x[-1])
        | (pos[:, 1] < y[0])
        | (pos[:, 1] > y[-1])
    ):
        raise ValueError("Phone outside map extent")

    ix = np.abs(
        x[None, :] - pos[:, 0, None]
    ).argmin(axis=1)

    iy = np.abs(
        y[None, :] - pos[:, 1, None]
    ).argmin(axis=1)

    if not np.all(mask[iy, ix]):
        raise ValueError(
            "Phone outside audience region"
        )

    return maps[:, iy, ix]


def prepare_interpolation_data(
    sound_maps,
    x,
    y,
    phone_counts,
    layouts,
    seed=42,
    allowed_mask=None
):
    """Prepare microphone measurements and reference sound levels."""

    x, y, mask, _ = _validate(
        x, y, allowed_mask
    )

    maps = np.asarray(sound_maps, dtype=float)

    if maps.ndim == 2:
        maps = maps[None, :, :]

    if maps.ndim != 3 or maps.shape[1:] != mask.shape:
        raise ValueError(
            "sound_maps dimensions do not match x/y"
        )

    X, Y = np.meshgrid(x, y)

    query = np.column_stack((
        X[mask],
        Y[mask]
    ))

    truth = maps[:, mask]

    result = {}

    for layout in layouts:

        for n in phone_counts:

            positions = generate_phone_positions(
                x, y, n, layout, seed, mask
            )

            measured = extract_measurements(
                maps, x, y, positions, mask
            )

            result[(n, layout)] = {
                "phone_positions": positions,
                "phone_levels": measured,
                "query_positions": query,
                "true_levels": truth,
                "allowed_mask": mask
            }

    return result