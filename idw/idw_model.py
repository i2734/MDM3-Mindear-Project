import numpy as np


def idw(phone_positions, phone_levels, query_positions, p=2.0, k=None):
    """Inverse distance weighted interpolation of sound levels.

    phone_positions: (N, 2)   phone coordinates
    phone_levels:    (T, N)   measured levels in dB (one row per time step)
    query_positions: (M, 2)   points to predict at
    p: distance power (higher = more local)
    k: if given, only the k nearest phones are used for each point, otherwise all phones are used, decides how many phones each point on the map is allowed to listen to when IDW makes its prediction there

    Returns predicted_levels: (T, M) in dB. Levels are averaged in energy
    (10**(L/10)), then converted back to dB.
    """
    # distance from every prediction point to every phone: (M, N)
    d = np.linalg.norm(
        query_positions[:, None, :] - phone_positions[None, :, :], axis=2
    )

    # closer phones get larger weights (tiny constant avoids dividing by zero if the point is exactly on top of a phone)
    w = 1.0 / (d + 1e-9) ** p

    # optionally keep only the k nearest phones for each prediction point
    if k is not None and k < d.shape[1]:
        cutoff = np.sort(d, axis=1)[:, k - 1:k]
        w = np.where(d <= cutoff, w, 0.0)

    w = w / w.sum(axis=1, keepdims=True)      # each row sums to 1

    energy = 10.0 ** (phone_levels / 10.0)    # dB -> energy, (T, N)
    return 10.0 * np.log10(energy @ w.T)      # weighted average -> dB, (T, M)


def choose_p_k(phone_positions, phone_levels,
               p_values=(1, 2, 3, 4), k_values=(None, 2, 4, 8, 16)):
    """Pick p and k using only the phone readings (leave one out).

    Each phone is hidden in turn and predicted from the other phones
    The (p, k) with the smallest error on these hidden phones is returned
    """
    N = len(phone_positions)
    if N < 3:
        return 2.0, None                      # too few phones to test, use default

    best, best_error = (2.0, None), np.inf
    for p in p_values:
        for k in k_values:
            if k is not None and k >= N:
                continue                      # would use every phone anyway
            errors = []
            for i in range(N):
                others = np.arange(N) != i    # all phones except phone i
                pred = idw(phone_positions[others], phone_levels[:, others],
                           phone_positions[i:i + 1], p=p, k=k)
                errors.append(pred[:, 0] - phone_levels[:, i])
            rmse = np.sqrt(np.mean(np.square(errors)))
            if rmse < best_error:
                best, best_error = (p, k), rmse
    return best
