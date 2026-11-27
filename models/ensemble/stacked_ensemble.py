"""Stacked ensemble: a weighted blend of the tuned GP, IDW (dB) and diffusion LMS.

Each base model has different strengths (the tuned GP is the most accurate overall,
IDW in dB is simple and robust to badly-calibrated phones, diffusion LMS copes best
with clustered phones on large open venues). The ensemble predicts with all three and
blends their maps:

    prediction = w_gp * GP + w_idw * IDW + w_diffusion * diffusion,
    weights >= 0 and summing to 1

The weights are learnt from the phones only, never from the true map:
  1. The phones are split into 5 groups (folds). Each base model is fitted on 4 groups
     and predicts the readings of the 5th, so every phone gets an out-of-sample
     prediction from every model.
  2. The weights are the blend (searched in steps of 0.05) whose out-of-sample
     predictions are closest to what the phones actually read (smallest mean
     squared error over all phones and time steps).
  3. All three models are refitted on every phone and blended with those weights.

A base model whose final map contains impossible values (not finite, or at or below
0 dB) is dropped and the other weights rescaled, so one model failing doesn't spoil
the blend.

    from stacked_ensemble import StackedEnsemble
    model = StackedEnsemble()
    pred = model.fit_predict(phone_positions, phone_levels, query_positions)
    model.weights_          # e.g. {"gp": 0.6, "idw": 0.3, "diffusion": 0.1}

Inputs: phone_positions (N, 2) in m, phone_levels (T, N) in dB, query_positions (M, 2).
Output: predicted levels (T, M) in dB.
"""

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for folder in ("models/idw", "models/gaussian_process", "models/diffusion_lms"):
    if str(ROOT / folder) not in sys.path:
        sys.path.insert(0, str(ROOT / folder))

from diffusion_lms_model import DiffusionAcousticLMS    # noqa: E402
from gp_tuned import predict_gp_tuned                    # noqa: E402
from idw_model import idw, choose_p_k                    # noqa: E402

# diffusion settings used in the model comparison (diffusion_fianl_testing.ipynb)
DIFFUSION_SETTINGS = dict(n_basis=100, basis_width_ratio=0.08, learning_rate=0.15,
                          n_iterations=900, communication_radius=None,
                          warm_start=True, readout="mean")


def _gp(pos, lev, q):
    return predict_gp_tuned(pos, lev, q)[0]


def _idw_db(pos, lev, q):
    p, k = choose_p_k(pos, lev, domain="db")
    return idw(pos, lev, q, p=p, k=k, domain="db")


def _diffusion(pos, lev, q):
    return DiffusionAcousticLMS(**DIFFUSION_SETTINGS).fit_predict(pos, lev, q)


BASE_MODELS = {"gp": _gp, "idw": _idw_db, "diffusion": _diffusion}


def _valid(pred):
    return bool(np.all(np.isfinite(pred)) and np.all(pred > 0))


def _simplex_grid(n_models, step):
    """Every weight combination on a grid of `step` that is >= 0 and sums to 1."""
    n = int(round(1 / step))
    for combo in itertools.product(range(n + 1), repeat=n_models - 1):
        if sum(combo) <= n:
            yield np.array(list(combo) + [n - sum(combo)]) / n


class StackedEnsemble:
    def __init__(self, n_folds=5, weight_step=0.05, seed=0, base_models=None):
        self.n_folds = n_folds
        self.weight_step = weight_step
        self.seed = seed
        self.base_models = base_models or BASE_MODELS
        self.weights_ = None

    def _out_of_fold(self, pos, lev):
        """Each model's prediction at every phone, made without that phone's group."""
        N = len(pos)
        folds = np.random.default_rng(self.seed).permutation(N) % min(self.n_folds, N)
        oof = {name: np.full(lev.shape, np.nan) for name in self.base_models}
        for f in range(folds.max() + 1):
            test, train = folds == f, folds != f
            if train.sum() < 2:
                continue
            for name, model in self.base_models.items():
                try:
                    oof[name][:, test] = model(pos[train], lev[:, train], pos[test])
                except np.linalg.LinAlgError:
                    pass                      # leaves NaN -> this model gets no weight
        return oof

    def fit_predict(self, phone_positions, phone_levels, query_positions):
        pos = np.asarray(phone_positions, dtype=float)
        lev = np.atleast_2d(np.asarray(phone_levels, dtype=float))
        q = np.asarray(query_positions, dtype=float)
        names = list(self.base_models)

        # 1-2. learn the blend from out-of-sample predictions at the phones
        oof = self._out_of_fold(pos, lev)
        usable = [n for n in names if _valid(oof[n])]
        if not usable:                         # fall back to IDW, which can't blow up
            usable = ["idw"] if "idw" in names else names[:1]
        stacked = np.stack([oof[n] for n in usable])          # (models, T, N)
        best_w, best_err = None, np.inf
        for w in _simplex_grid(len(usable), self.weight_step):
            err = np.nanmean((np.tensordot(w, stacked, axes=1) - lev) ** 2)
            if err < best_err:
                best_w, best_err = w, err

        # 3. refit every used model on all phones and blend
        preds = {}
        for n, w in zip(usable, best_w):
            if w > 0:
                p = self.base_models[n](pos, lev, q)
                if _valid(p):
                    preds[n] = (w, p)
        if not preds:                          # every weighted model failed
            preds = {"idw": (1.0, self.base_models["idw"](pos, lev, q))}
        total = sum(w for w, _ in preds.values())
        self.weights_ = {n: round(preds[n][0] / total, 3) if n in preds else 0.0 for n in names}
        return sum((w / total) * p for w, p in preds.values())
