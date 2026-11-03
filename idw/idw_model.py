import numpy as np


def idw(phone_positions, phone_levels, query_positions, p=2.0, k=None):
    """Inverse distance weighted interpolation of sound levels

    phone_positions: (N, 2) phone coordinates in meters
    phone_levels: (T, N) measured levels in dB (one row per time step)
    query_positions: (M, 2) points to predict at
    p: distance power (higher = more local, nearest phones dominate, low = more global, all phones contribute)
    k: if given, only the k nearest phones are used for each point

    Returns predicted_levels: (T, M) in dB, levels are averaged in energy (10**(L/10)), then converted back to dB
    """
    # distance from every query point to every phone: (M, N)
    d = np.linalg.norm(query_positions[:, None, :] - phone_positions[None, :, :], axis=2)

    # closer phones get larger weights (tiny constant avoids dividing by zero if the query point is exactly at a phone position)
    w = 1.0 / (d + 1e-9) ** p

    # optionally keep only the k nearest phones for each query point
    if k is not None and k < d.shape[1]:
        cutoff = np.sort(d, axis=1)[:, k - 1:k]
        w = np.where(d <= cutoff, w, 0.0)

    w = w / w.sum(axis=1, keepdims=True)      # each row sums to 1

    energy = 10.0 ** (phone_levels / 10.0)    # dB -> energy, (T, N), we have to do this because dB is logarithmic and we want to average in linear
    return 10.0 * np.log10(energy @ w.T)      # weighted average -> dB, (T, M)
