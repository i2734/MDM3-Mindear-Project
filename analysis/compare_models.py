"""Comparing the four interpolation models on the same data

Every model gets exactly the same phone positions, phone readings and time frames, and is scored only at audience points that do not have a phone on them.
Results are written to one CSV (one row per model/venue/phone count/layout/seed) so every table and figure in the report can be made from the same file.

Run from the repo root, for example:
    python3 analysis/compare_models.py                            # perfect phone readings
    python3 analysis/compare_models.py --readings real_phones     # readings with phone error
    python3 analysis/compare_models.py --venues concert --seeds 0 --models idw kriging

Phone readings (PHONE_ERROR in sound_maps/venues/bias_and_variance_extract_sound_venues.py):
    perfect     - true levels at the phones
    real_phones - each phone has a fixed offset taken from the calibration experiment,
                  plus 3.2 dB random noise on every reading
    legacy      - the earlier setting (-2.5914 dB on every reading, 1.8874 dB noise)

Models (settings as used in the group's own testing):
    idw       - models/idw/idw_model.py, p and k tuned by leave one out on the phones,
                averaging readings as energy
    idw_db    - same, averaging readings in dB
    kriging   - models/kriging/Ordinary_Kriging_Model_no_buffer.py (no safety buffer),
                averaging readings as energy
    kriging_db - same kriging in dB (Ordinary_Kriging_Model_buffer_diffusion.py with
                domain="db" and no safety margin or peak boost)
    gp        - models/gaussian_process/gp_all_model.py, fixed RBF kernel (15 m length scale)
    gp_tuned  - same GP, length scale and noise level chosen by leave one out on the phones
                (models/gaussian_process/gp_tuned.py)
                (the tuned GP from the gp_working_final notebooks is not in the repo)
    ensemble  - models/ensemble/stacked_ensemble.py: weighted blend of gp_tuned, idw_db and
                diffusion, weights learnt by 5-fold cross-validation on the phones
    diffusion - models/diffusion_lms/diffusion_lms_model.py, settings from
                diffusion_fianl_testing.ipynb
"""

import argparse
import functools
import csv
import importlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ("sound_maps/venues", "models/idw", "models/kriging",
               "models/gaussian_process", "models/diffusion_lms", "models/ensemble"):
    sys.path.insert(0, str(ROOT / folder))

VENUES = {                       # venue -> (sound map module, frame stride)
    "concert": ("final_concert_map", 1),      # 1 frame (static)
    "stadium": ("final_stadium_map", 5),      # 91 frames -> every 5th (19 frames)
    "f1": ("final_race_track_map", 20),       # 721 frames -> every 20th (37 frames)
}
PHONE_COUNTS = [10, 30, 50, 100, 250]
LAYOUTS = ["random", "grid", "clustered"]
READINGS = ["perfect", "real_phones", "legacy"]   # phone_error settings
DEFAULT_MODELS = ["idw", "idw_db", "kriging", "kriging_db", "gp_tuned", "diffusion", "ensemble"]

DIFF_SETTINGS = dict(n_basis=100, basis_width_ratio=0.08, learning_rate=0.15,
                        n_iterations=900, communication_radius=None,
                        warm_start=True, readout="mean")


# ---------------------------------------------------------------- models
def run_idw(pos, lev, q, domain="energy"):
    from idw_model import idw, choose_p_k
    p, k = choose_p_k(pos, lev, domain=domain)
    return idw(pos, lev, q, p=p, k=k, domain=domain), f"p={p:g},k={'all' if k is None else k}"


def run_idw_db(pos, lev, q):
    return run_idw(pos, lev, q, domain="db")


def run_kriging(pos, lev, q):
    from Ordinary_Kriging_Model_no_buffer import ExactAcousticKriging
    return ExactAcousticKriging().fit_predict(pos, lev, q), ""


def run_kriging_db(pos, lev, q):
    from Ordinary_Kriging_Model_buffer_diffusion import ExactAcousticKriging
    model = ExactAcousticKriging(safety_margin_db=0.0, peak_boost=False, domain="db")
    return model.fit_predict(pos, lev, q), ""


def run_gp(pos, lev, q):
    from gp_all_model import predict_gp, LENGTH_SCALE_M
    return predict_gp(pos, lev, q), f"length_scale={LENGTH_SCALE_M:g}m"


def run_gp_tuned(pos, lev, q):
    from gp_tuned import predict_gp_tuned
    pred, (ell, noise) = predict_gp_tuned(pos, lev, q)
    return pred, f"length_scale={ell:.1f}m,noise={noise:g}dB"


def run_diffusion(pos, lev, q):
    from diffusion_lms_model import DiffusionAcousticLMS
    return DiffusionAcousticLMS(**DIFF_SETTINGS).fit_predict(pos, lev, q), ""


def run_ensemble(pos, lev, q):
    from stacked_ensemble import StackedEnsemble
    model = StackedEnsemble()
    pred = model.fit_predict(pos, lev, q)
    return pred, ",".join(f"w_{k}={v:g}" for k, v in model.weights_.items())


MODELS = {"idw": run_idw, "idw_db": run_idw_db, "kriging": run_kriging,
          "kriging_db": run_kriging_db, "gp": run_gp, "gp_tuned": run_gp_tuned,
          "diffusion": run_diffusion, "ensemble": run_ensemble}


# ---------------------------------------------------------------- scoring
def score(pred, true, ev):
    """Errors at non-phone audience points over all frames (prediction - truth, dB)."""
    pred, true = pred[:, ev], true[:, ev]
    valid = np.isfinite(pred) & (pred > 0)          # kriging can return -120 dB
    err = pred - true
    under = -err                                    # positive = model reads too quiet
    return {
        "rmse_db": np.sqrt(np.mean(err ** 2)),
        "mae_db": np.mean(np.abs(err)),
        "bias_db": np.mean(err),
        "under_p95_db": np.percentile(under, 95),
        "max_under_db": np.max(under),
        "within_3db": np.mean(np.abs(err) <= 3),
        "n_invalid": int(np.size(pred) - valid.sum()),
    }


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readings", choices=READINGS, default="perfect")
    ap.add_argument("--venues", nargs="+", choices=VENUES, default=list(VENUES))
    ap.add_argument("--models", nargs="+", choices=MODELS, default=DEFAULT_MODELS)
    ap.add_argument("--counts", nargs="+", type=int, default=PHONE_COUNTS)
    ap.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=LAYOUTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="with perfect readings grid layouts don't depend on the seed, so only the first seed is used for grid")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or ROOT / "analysis" / "results" / f"model_comparison_{args.readings}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module("bias_and_variance_extract_sound_venues")
    prepare = functools.partial(module.prepare_interpolation_data, phone_error=args.readings)

    # resume: skip rows already in the output file
    done = set()
    if out.exists():
        with out.open(newline="") as f:
            done = {(r["venue"], r["model"], r["phone_count"], r["layout"], r["seed"])
                    for r in csv.DictReader(f)}
    fields = ["readings", "venue", "model", "phone_count", "layout", "seed", "frames",
              "rmse_db", "mae_db", "bias_db", "under_p95_db", "max_under_db",
              "within_3db", "n_invalid", "runtime_s", "settings"]
    new_file = not out.exists()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for venue in args.venues:
            module, stride = VENUES[venue]
            maps, x, y, mask = importlib.import_module(module).load_sound_map()
            maps = np.asarray(maps)
            if maps.ndim == 2:
                maps = maps[None]
            maps = maps[::stride]
            for seed in args.seeds:
                # with perfect readings a grid layout has nothing random in it, so it only
                # needs one seed; with phone error the offsets and noise change per seed
                layouts = [l for l in args.layouts
                           if l != "grid" or seed == args.seeds[0] or args.readings != "perfect"]
                if not layouts:
                    continue
                data = prepare(maps, x, y, args.counts, layouts, seed=seed, allowed_mask=mask)
                for (count, layout), d in data.items():
                    for name in args.models:
                        key = (venue, name, str(count), layout, str(seed))
                        if key in done:
                            continue
                        t0 = time.perf_counter()
                        pred, settings = MODELS[name](d["phone_positions"], d["phone_levels"],
                                                      d["query_positions"])
                        row = dict(readings=args.readings, venue=venue, model=name,
                                   phone_count=count, layout=layout, seed=seed,
                                   frames=len(maps), runtime_s=time.perf_counter() - t0,
                                   settings=settings,
                                   **score(pred, d["true_levels"], d["evaluation_mask"]))
                        writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                                         for k, v in row.items()})
                        f.flush()
                        print(f"{venue:8s} {name:9s} {count:3d} {layout:9s} seed {seed}: "
                              f"RMSE {row['rmse_db']:.2f} dB  ({row['runtime_s']:.1f}s)", flush=True)
    print("Saved", out)


if __name__ == "__main__":
    main()
