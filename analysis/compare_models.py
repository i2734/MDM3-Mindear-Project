"""Comparing the four interpolation models on the same data

Every model gets exactly the same phone positions, phone readings and time frames, and is scored only at audience points that do not have a phone on them.
Results are written to one CSV (one row per model/venue/phone count/layout/seed) so every table and figure in the report can be made from the same file.

Run from the repo root, for example:
    python3 analysis/compare_models.py                       # perfect phone readings
    python3 analysis/compare_models.py --readings noisy      # readings with phone error
    python3 analysis/compare_models.py --venues concert --seeds 0 --models idw kriging

Phone readings:
    perfect  - true levels at the phones (sound_maps/venues/final_extract_sound_venues.py)
    noisy    - readings with simulated phone error
               (sound_maps/venues/bias_and_variance_extract_sound_venues.py)

Models (settings as used in the group's own testing):
    idw       - models/idw/idw_model.py, p and k tuned by leave one out on the phones
    kriging   - models/kriging/Ordinary_Kriging_Model_no_buffer.py (no safety buffer)
    gp        - models/gaussian_process/gp_all_model.py, fixed RBF kernel (15 m length scale)
    gp_tuned  - same GP, length scale chosen by leave one out on the phones
                (the tuned GP from the gp_working_final notebooks is not in the repo)
    diffusion - models/diffusion_lms/diffusion_lms_model.py, settings from
                diffusion_fianl_testing.ipynb
"""

import argparse
import csv
import importlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ("sound_maps/venues", "models/idw", "models/kriging",
               "models/gaussian_process", "models/diffusion_lms"):
    sys.path.insert(0, str(ROOT / folder))

VENUES = {                       # venue -> (sound map module, frame stride)
    "concert": ("final_concert_map", 1),      # 1 frame (static)
    "stadium": ("final_stadium_map", 5),      # 91 frames -> every 5th (19 frames)
    "f1": ("final_race_track_map", 20),       # 721 frames -> every 20th (37 frames)
}
PHONE_COUNTS = [10, 30, 50, 100, 250]
LAYOUTS = ["random", "grid", "clustered"]
READINGS = {"perfect": "final_extract_sound_venues",
            "noisy": "bias_and_variance_extract_sound_venues"}

DIFF_SETTINGS = dict(n_basis=100, basis_width_ratio=0.08, learning_rate=0.15,
                        n_iterations=900, communication_radius=None,
                        warm_start=True, readout="mean")


# ---------------------------------------------------------------- models
def run_idw(pos, lev, q):
    from idw_model import idw, choose_p_k
    p, k = choose_p_k(pos, lev)
    return idw(pos, lev, q, p=p, k=k), f"p={p:g},k={'all' if k is None else k}"


def run_kriging(pos, lev, q):
    from Ordinary_Kriging_Model_no_buffer import ExactAcousticKriging
    return ExactAcousticKriging().fit_predict(pos, lev, q), ""


def run_gp(pos, lev, q):
    from gp_all_model import predict_gp, LENGTH_SCALE_M
    return predict_gp(pos, lev, q), f"length_scale={LENGTH_SCALE_M:g}m"


def run_gp_tuned(pos, lev, q):
    """Same GP, but the length scale is chosen by leave-one-out on the phones only
    (like IDW's p and k), from a range scaled to the size of the venue."""
    import gp_all_model as gp
    domain = np.vstack((pos, q))
    diagonal = np.linalg.norm(domain.max(axis=0) - domain.min(axis=0))
    candidates = diagonal * np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
    centred = (lev - lev.mean(axis=1, keepdims=True)).T          # (N, T)
    original = gp.LENGTH_SCALE_M
    best, best_err = original, np.inf
    try:
        for ell in candidates:
            gp.LENGTH_SCALE_M = ell
            K = gp.rbf_kernel(pos, pos) + (gp.NOISE_STD_DB**2 + gp.JITTER) * np.eye(len(pos))
            K_inv = np.linalg.inv(K)
            # closed-form leave-one-out residuals for every phone and frame
            loo = (K_inv @ centred) / np.diag(K_inv)[:, None]
            err = np.mean(loo**2)
            if err < best_err:
                best, best_err = ell, err
        gp.LENGTH_SCALE_M = best
        pred = gp.predict_gp(pos, lev, q)
    finally:
        gp.LENGTH_SCALE_M = original
    return pred, f"length_scale={best:.1f}m"


def run_diffusion(pos, lev, q):
    from diffusion_lms_model import DiffusionAcousticLMS
    return DiffusionAcousticLMS(**DIFF_SETTINGS).fit_predict(pos, lev, q), ""


MODELS = {"idw": run_idw, "kriging": run_kriging, "gp": run_gp, "gp_tuned": run_gp_tuned,
          "diffusion": run_diffusion}


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
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--counts", nargs="+", type=int, default=PHONE_COUNTS)
    ap.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=LAYOUTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="grid layouts don't depend on the seed, so only the first seed is used for grid")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or ROOT / "analysis" / "results" / f"model_comparison_{args.readings}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    prepare = importlib.import_module(READINGS[args.readings]).prepare_interpolation_data

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
                layouts = [l for l in args.layouts if l != "grid" or seed == args.seeds[0]]
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
