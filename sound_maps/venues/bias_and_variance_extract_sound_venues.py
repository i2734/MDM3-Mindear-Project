"""Place simulated phones in the venue and sample sound levels at their locations."""

import numpy as np


# Phone measurement error added to the simulated readings (phone_error argument below).
#   perfect     - no error: phones read the true level
#   real_phones - each simulated phone gets one fixed offset, picked at random from the
#                 mean errors of the six phones in the calibration experiment
#                 (calibration/SoundTracker_Calibration_Kit/03_Analysis/Summary_By_Phone.csv:
#                 IP01-IP04, SA01, SA02), plus random noise of SD 3.2 dB on every reading
#                 (the average per-phone SD in the same file)
#   legacy      - the earlier setting: every reading -2.5914 dB plus noise of SD 1.8874 dB
PHONE_ERROR = {
    "perfect": None,
    "real_phones": {"phone_offsets_db": [-13.93, -9.92, -11.05, -12.57, -32.44, 3.05],
                    "noise_sd_db": 3.2},
    "legacy": {"phone_offsets_db": [-2.5914], "noise_sd_db": 1.8874},
}


def add_phone_error(measured, rng, error):
    """Add measurement error to the true levels at the phones, measured shape (T, N).

    Each phone gets a fixed offset (the same at every time step) plus random noise
    that changes every reading.
    """
    if error is None:
        return measured
    T, N = measured.shape
    choices = error["phone_offsets_db"]
    if len(choices) == 1:
        offsets = np.full(N, choices[0])                          # same offset for every phone
    else:
        offsets = rng.choice(choices, size=N)                     # one fixed offset per phone
    noise = rng.normal(0.0, error["noise_sd_db"], size=(T, N))    # new noise every reading
    return measured + offsets[None, :] + noise


def _validate(x, y, allowed_mask):
    """Find the grid coordinates where phones are allowed to be placed."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # if no audience mask is given, allow phones anywhere on the grid
    mask = np.ones((len(y), len(x)), bool) if allowed_mask is None else np.asarray(allowed_mask, dtype=bool)

    # get the coordinates of all the positions where phones can be placed
    X, Y = np.meshgrid(x, y)
    candidates = np.column_stack((X[mask], Y[mask]))

    return x, y, mask, candidates


def _spread_select(points, n):
    """Pick n points by repeatedly choosing the one furthest from any already picked."""
    # start with the point closest to the centre
    chosen = np.empty(n, dtype=int)
    chosen[0] = np.argmin(np.sum((points - points.mean(axis=0))**2, axis=1))

    nearest = np.sum((points - points[chosen[0]])**2, axis=1)
    nearest[chosen[0]] = -1

    # keep choosing the point furthest from its nearest selected phone
    for i in range(1, n):
        chosen[i] = np.argmax(nearest)
        new = np.sum((points - points[chosen[i]])**2, axis=1)
        nearest = np.minimum(nearest, new)
        nearest[chosen[:i + 1]] = -1

    return points[chosen]


def generate_phone_positions(x, y, n_phones, layout, seed=42, allowed_mask=None):
    """Choose n phone locations using random, clustered or roughly even grid spacing."""
    x, y, mask, candidates = _validate(x, y, allowed_mask)
    rng = np.random.default_rng(seed)

    # random phone positions across the audience area
    if layout == "random":
        indices = rng.choice(len(candidates), n_phones, replace=False)
        return candidates[indices]

    # clustered phone positions
    if layout == "clustered":

        # randomly choose a centre, then place phones more closely around it
        centre = candidates[rng.integers(len(candidates))]
        #scale the cluster size so it is concentrated to 10% of the venue width and height, minimum of 1m (design choice)
        scales = np.maximum([np.ptp(x) / 10, np.ptp(y) / 10], 1e-12)

        # add Gumbel noise so phones are randomly selected, with positions closer to the cluster centre more likely to be chosen
        # uses "Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement" source
        score = -0.5 * np.sum(((candidates - centre) / scales)**2, axis=1) + rng.gumbel(size=len(candidates))
        indices = np.argpartition(score, -n_phones)[-n_phones:]

        return candidates[indices]

    # grid phone positions
    if layout == "grid":

        xmin, xmax = x[0], x[-1]
        ymin, ymax = y[0], y[-1]
        best = None
        best_score = float("inf")

        # estimate the required grid size and double it to account for the irregular audience area
        max_side = min(max(len(x), len(y)), max(2, int(2 * np.sqrt(n_phones / mask.mean()))))
        minimum = 2 if n_phones >= 4 else 1

        for rows in range(minimum, max_side + 1):
            for cols in range(minimum, max_side + 1):

                if rows * cols < n_phones:
                    continue

                # equally spaced points across the full venue
                gx = np.linspace(xmin, xmax, cols + 2)[1:-1]
                gy = np.linspace(ymin, ymax, rows + 2)[1:-1]
                xx, yy = np.meshgrid(gx, gy)

                # move the points onto the nearest sound map grid coordinates
                ix = np.abs(x[None, :] - xx.ravel()[:, None]).argmin(axis=1)
                iy = np.abs(y[None, :] - yy.ravel()[:, None]).argmin(axis=1)

                # remove points outside the audience area and any duplicates
                keep = mask[iy, ix]
                flat = np.unique(iy[keep] * len(x) + ix[keep])

                if len(flat) < n_phones:
                    continue

                points = np.column_stack((x[flat % len(x)], y[flat // len(x)]))

                dx = (xmax - xmin) / (cols + 1)
                dy = (ymax - ymin) / (rows + 1)

                # prefer similar spacing in both directions and fewer extra points
                spacing_penalty = abs(np.log(max(dx, 1e-12) / max(dy, 1e-12)))
                # penalise grids with extra points and unequal spacing, with a weighting of 0.5 for the spacing penalty
                score = len(points) - n_phones + 0.5 * spacing_penalty

                if score < best_score:
                    best_score = score
                    best = points

        # if there are extra grid points, keep the most spread out ones
        if len(best) == n_phones:
            return best

        return _spread_select(best, n_phones)


def extract_measurements(sound_maps, x, y, phone_positions, allowed_mask=None):
    """Read each phone's sound level from the nearest map cell, for every time frame."""
    x, y, mask, _ = _validate(x, y, allowed_mask)
    maps = np.asarray(sound_maps, dtype=float)

    # add a time dimension if there is only one sound map
    if maps.ndim == 2:
        maps = maps[None, :, :]

    pos = np.asarray(phone_positions, dtype=float)

    # find the nearest sound map grid point to each phone
    ix = np.abs(x[None, :] - pos[:, 0, None]).argmin(axis=1)
    iy = np.abs(y[None, :] - pos[:, 1, None]).argmin(axis=1)

    # extract the sound level at each phone for every time frame
    return maps[:, iy, ix]


def prepare_interpolation_data(sound_maps, x, y, phone_counts, layouts, seed=42, allowed_mask=None,
                               phone_error="real_phones"):   
    """Build phone measurements and full-map true values for each layout and phone count.

    phone_error: "perfect", "real_phones" or "legacy" (see PHONE_ERROR at the top).
    """
    x, y, mask, _ = _validate(x, y, allowed_mask)    
    maps = np.asarray(sound_maps, dtype=float)

    if maps.ndim == 2:
        maps = maps[None, :, :]

    # all the audience positions where we want to estimate sound levels
    X, Y = np.meshgrid(x, y)
    query = np.column_stack((X[mask], Y[mask]))

    # actual simulated sound levels used to compare with interpolation predictions
    truth = maps[:, mask]

    result = {}

    # generate each phone layout and extract its measurements
    for layout in layouts:
        for n in phone_counts:
            positions = generate_phone_positions(x, y, n, layout, seed, mask)
            measured = extract_measurements(maps, x, y, positions, mask)

            rng = np.random.default_rng(seed + n + sum(map(ord, layout)))
            # add phone measurement error (see PHONE_ERROR at the top)
            measured = add_phone_error(measured, rng, PHONE_ERROR[phone_error])
            # Find the grid indices occupied by microphones
            ix = np.abs(x[None, :] - positions[:, 0, None]).argmin(axis=1)
            iy = np.abs(y[None, :] - positions[:, 1, None]).argmin(axis=1)

            training_flat = iy * len(x) + ix

            # Identify query locations that are NOT microphone positions
            query_flat = np.flatnonzero(mask.ravel())
            evaluation_mask = ~np.isin(query_flat, training_flat)

            result[(n, layout)] = {
                "phone_positions": positions,
                "phone_levels": measured,
                "query_positions": query,
                "true_levels": truth,
                "evaluation_mask": evaluation_mask
                    }

    return result
