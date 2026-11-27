"""Maps of each model's prediction against the simulated ground truth.

Every model is given exactly the same phones and readings at the same time step, so
the maps can be compared directly. For each phone setting it draws:
  - one figure per model: a row per venue, showing truth | prediction | error
  - one figure per venue: all models side by side (maps on top, errors below)

    python3 analysis/plot_example_maps.py                                   # the dB models
    python3 analysis/plot_example_maps.py --models ensemble --tag part3_ensemble --no-venue-figures

Phone settings (default: all three), each with 50 phones and seed 0:
    random_perfect, clustered_perfect, random_real_phones
Writes analysis/results/figures/<tag>/maps_<model or venue>_<layout>_<readings>.pdf/.png
"""

import argparse
import importlib
import inspect
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import compare_models as cm                      # noqa: E402  (also sets up import paths)
from plot_comparison import STYLE               # noqa: E402

FIGS = ROOT / "analysis" / "results" / "figures"
VENUES = [("concert", "Concert"), ("stadium", "Stadium"), ("f1", "F1 racetrack")]
SETTINGS = {"random_perfect": ("random", "perfect"),
            "clustered_perfect": ("clustered", "perfect"),
            "random_real_phones": ("random", "real_phones")}
DB_MODELS = ["idw_db", "kriging_db", "gp_tuned", "diffusion"]
MAP_CMAP, ERR_CMAP = "Oranges", "RdBu_r"


def label(model):
    return STYLE[model][0] if model in STYLE else model


def pick_frame(maps, venue):
    """A representative time step: the loudest moment (stadium), the group's F1 frame."""
    if maps.shape[0] == 1:
        return 0
    if venue == "f1":
        return 80
    return int(np.nanargmax(np.nanmean(maps.reshape(len(maps), -1), axis=1)))


def to_grid(values, mask):
    img = np.full(mask.shape, np.nan)
    img[mask] = values
    return img


def draw_outline(module, ax, frame):
    if not hasattr(module, "draw_venue"):
        return
    params = inspect.signature(module.draw_venue).parameters
    kwargs = {k: v for k, v in (("frame", frame), ("show_cars", True)) if k in params}
    try:
        module.draw_venue(ax, **kwargs)
    except Exception:
        pass                                      # the outline is only decoration


def show(ax, img, extent, module, frame, cmap, vmin, vmax, phones=None, title=""):
    im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    draw_outline(module, ax, frame)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    if phones is not None:
        ax.scatter(phones[:, 0], phones[:, 1], s=5, c="#2b2b2b", linewidths=0, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8.5)
    return im


def compute(models, count, seed):
    """Predictions of every model for every venue and phone setting (same phones)."""
    prepare = importlib.import_module(
        "bias_and_variance_extract_sound_venues").prepare_interpolation_data
    out = {}
    for venue, _ in VENUES:
        module = importlib.import_module(cm.VENUES[venue][0])
        maps, x, y, mask = module.load_sound_map()
        maps = np.asarray(maps)
        maps = maps[None] if maps.ndim == 2 else maps
        mask = np.asarray(mask, dtype=bool)
        frame = pick_frame(maps, venue)
        for key, (layout, readings) in SETTINGS.items():
            d = prepare(maps[frame:frame + 1], x, y, [count], [layout], seed=seed,
                        allowed_mask=mask, phone_error=readings)[(count, layout)]
            preds = {}
            for m in models:
                p, _ = cm.MODELS[m](d["phone_positions"], d["phone_levels"], d["query_positions"])
                preds[m] = np.asarray(p)[0]
                print(f"{venue:8s} {key:19s} {m}", flush=True)
            out[(venue, key)] = dict(module=module, x=x, y=y, mask=mask, frame=frame,
                                     truth=d["true_levels"][0], phones=d["phone_positions"],
                                     ev=d["evaluation_mask"], preds=preds)
    return out


def scales(r, models):
    """Shared colour range for truth and predictions, and a symmetric error range."""
    vals = np.concatenate([r["truth"]] + [r["preds"][m] for m in models])
    lo, hi = np.nanpercentile(vals, [1, 99.5])
    errs = np.concatenate([r["preds"][m] - r["truth"] for m in models])
    lim = max(3.0, np.nanpercentile(np.abs(errs), 98))
    return lo, hi, lim


def rmse(r, m):
    e = (r["preds"][m] - r["truth"])[r["ev"]]
    return np.sqrt(np.mean(e ** 2))


def describe(key, count):
    layout, readings = SETTINGS[key]
    return f"{count} {layout} phones, {'perfect' if readings == 'perfect' else 'real'} phones"


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)
    print("Saved", path.with_suffix(".png"))


def model_figure(results, model, key, count, folder):
    """Rows = venues; columns = truth | prediction | error."""
    first = [results[(v, key)] for v, _ in VENUES]
    ratios = [np.ptp(r["y"]) / np.ptp(r["x"]) for r in first]
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 1.0 + 2.3 * sum(ratios)),
                             gridspec_kw=dict(height_ratios=ratios))
    for row, (venue, title) in enumerate(VENUES):
        r = results[(venue, key)]
        extent = [r["x"].min(), r["x"].max(), r["y"].min(), r["y"].max()]
        lo, hi, lim = scales(r, [model])
        panels = [(r["truth"], "Simulated truth", MAP_CMAP, lo, hi, None),
                  (r["preds"][model], label(model), MAP_CMAP, lo, hi, r["phones"]),
                  (r["preds"][model] - r["truth"], f"Error (RMSE {rmse(r, model):.2f} dB)",
                   ERR_CMAP, -lim, lim, r["phones"])]
        for col, (vals, ptitle, cmap, vmin, vmax, phones) in enumerate(panels):
            ax = axes[row, col]
            im = show(ax, to_grid(vals, r["mask"]), extent, r["module"], r["frame"], cmap,
                      vmin, vmax, phones, ptitle if (row == 0 or col == 2) else "")
            if col == 0:
                ax.set_ylabel(title, fontsize=10)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=7)
            cb.set_label("dB" if col < 2 else "prediction − truth (dB)", fontsize=7)
    fig.suptitle(f"{label(model)}: {describe(key, count)} (black dots = phones)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    layout, readings = SETTINGS[key]
    save(fig, folder / f"maps_{model}_{layout}_{readings}")


def venue_figure(results, models, venue, title, key, count, folder):
    """Top row: truth and every model's map. Bottom row: phones and every model's error."""
    r = results[(venue, key)]
    extent = [r["x"].min(), r["x"].max(), r["y"].min(), r["y"].max()]
    ratio = np.ptp(r["y"]) / np.ptp(r["x"])
    n = len(models) + 1
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n + 0.8, 2 * 2.0 * ratio + 1.2))
    lo, hi, lim = scales(r, models)
    im_map = show(axes[0, 0], to_grid(r["truth"], r["mask"]), extent, r["module"], r["frame"],
                  MAP_CMAP, lo, hi, title="Simulated truth")
    show(axes[1, 0], to_grid(r["truth"], r["mask"]), extent, r["module"], r["frame"],
         MAP_CMAP, lo, hi, r["phones"], title="Phone positions")
    for j, m in enumerate(models, start=1):
        show(axes[0, j], to_grid(r["preds"][m], r["mask"]), extent, r["module"], r["frame"],
             MAP_CMAP, lo, hi, r["phones"], title=label(m))
        im_err = show(axes[1, j], to_grid(r["preds"][m] - r["truth"], r["mask"]), extent,
                      r["module"], r["frame"], ERR_CMAP, -lim, lim, r["phones"],
                      title=f"Error, RMSE {rmse(r, m):.2f} dB")
    cb = fig.colorbar(im_map, ax=list(axes[0]), fraction=0.02, pad=0.01)
    cb.set_label("dB", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    cb = fig.colorbar(im_err, ax=list(axes[1]), fraction=0.02, pad=0.01)
    cb.set_label("prediction − truth (dB)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"{title}: {describe(key, count)} (black dots = phones)", fontsize=10)
    layout, readings = SETTINGS[key]
    save(fig, folder / f"maps_{venue}_{layout}_{readings}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DB_MODELS, choices=cm.MODELS)
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=SETTINGS)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="part1_model_maps", help="subfolder of figures/")
    ap.add_argument("--no-venue-figures", action="store_true")
    args = ap.parse_args()

    for key in list(SETTINGS):
        if key not in args.settings:
            del SETTINGS[key]
    results = compute(args.models, args.count, args.seed)
    folder = FIGS / args.tag
    for key in SETTINGS:
        for m in args.models:
            model_figure(results, m, key, args.count, folder)
        if not args.no_venue_figures:
            for venue, title in VENUES:
                venue_figure(results, args.models, venue, title, key, args.count, folder)


if __name__ == "__main__":
    main()
