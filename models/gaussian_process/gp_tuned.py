"""Gaussian process with its length scale and noise level tuned on the phones.

Uses the RBF GP in gp_all_model.py, but instead of a fixed 15 m length scale and
0.5 dB noise it picks both by leave-one-out on the phone readings only (like IDW's
p and k), so it adapts to each venue and to how noisy the phones are.

    from gp_tuned import predict_gp_tuned
    pred, (length_scale_m, noise_db) = predict_gp_tuned(phone_positions, phone_levels,
                                                        query_positions)

Inputs: phone_positions (N, 2) in m, phone_levels (T, N) in dB, query_positions (M, 2).
Output: predicted levels (T, M) in dB.
"""

import numpy as np

import gp_all_model as gp

LENGTH_SCALE_FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]   # of the venue diagonal
NOISE_LEVELS_DB = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def choose_hyperparameters(phone_positions, phone_levels, query_positions):
    """Length scale (m) and noise level (dB) with the smallest leave-one-out error."""
    domain = np.vstack((phone_positions, query_positions))
    diagonal = np.linalg.norm(domain.max(axis=0) - domain.min(axis=0))
    centred = (phone_levels - phone_levels.mean(axis=1, keepdims=True)).T      # (N, T)
    original = gp.LENGTH_SCALE_M
    best, best_err = (original, gp.NOISE_STD_DB), np.inf
    try:
        for fraction in LENGTH_SCALE_FRACTIONS:
            gp.LENGTH_SCALE_M = diagonal * fraction
            K_signal = gp.rbf_kernel(phone_positions, phone_positions)
            for noise in NOISE_LEVELS_DB:
                K = K_signal + (noise**2 + gp.JITTER) * np.eye(len(phone_positions))
                K_inv = np.linalg.inv(K)
                # closed-form leave-one-out residuals for every phone and frame
                loo = (K_inv @ centred) / np.diag(K_inv)[:, None]
                err = np.mean(loo**2)
                if err < best_err:
                    best, best_err = (gp.LENGTH_SCALE_M, noise), err
    finally:
        gp.LENGTH_SCALE_M = original
    return best


def predict_with(phone_positions, phone_levels, query_positions, length_scale_m, noise_db):
    """GP prediction with the given hyperparameters (restores the module defaults after)."""
    original = (gp.LENGTH_SCALE_M, gp.NOISE_STD_DB)
    try:
        gp.LENGTH_SCALE_M, gp.NOISE_STD_DB = length_scale_m, noise_db
        return gp.predict_gp(phone_positions, phone_levels, query_positions)
    finally:
        gp.LENGTH_SCALE_M, gp.NOISE_STD_DB = original


def predict_gp_tuned(phone_positions, phone_levels, query_positions):
    """Tune on the phones, then predict at the query points."""
    params = choose_hyperparameters(phone_positions, phone_levels, query_positions)
    return predict_with(phone_positions, phone_levels, query_positions, *params), params
