"""Make the report figures and summary table from compare_models.py output.

    python3 analysis/plot_comparison.py                       # uses model_comparison_perfect.csv
    python3 analysis/plot_comparison.py --readings real_phones
    python3 analysis/plot_comparison.py --models idw_db kriging_db gp_tuned diffusion --tag part2_db_comparison

Writes to analysis/results/:
    figures/rmse_vs_phones_<readings>.pdf/.png         RMSE vs phone count
    figures/under_p95_vs_phones_<readings>.pdf/.png    95th-percentile under-prediction
    figures/bias_removed_rmse_vs_phones_<readings>     RMSE once each map's average bias
                                                       is removed (only with phone error)
    figures/perfect_vs_real_phones.pdf/.png            effect of phone error, 50 random phones
    table_perfect_vs_real_phones.tex                   the same numbers as a LaTeX table
    summary_<readings>.csv                             mean and SD over seeds
    table_50_phones_<readings>.tex                     LaTeX table, 50 phones
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"

# model -> (label, colour, line style, marker, filled)
STYLE = {
    "idw":        ("IDW (energy)",          "#2a78d6", "-",  "o", True),
    "idw_db":     ("IDW (dB)",              "#2a78d6", "--", "o", False),
    "kriging":    ("Kriging (energy)",      "#eb6834", "-",  "s", True),
    "kriging_db": ("Kriging (dB)",          "#eb6834", "--", "s", False),
    "gp_tuned":   ("GP (tuned)",            "#1baf7a", "-",  "^", True),
    "gp":         ("GP (fixed 15 m)",       "#1baf7a", ":",  "^", False),
    "diffusion":  ("Diffusion LMS",         "#c98500", "-",  "D", True),
    "ensemble":   ("Stacked ensemble",      "#d4508a", "-",  "v", True),
}
READINGS_STYLE = {"perfect": ("Perfect phones", "#2a78d6"),
                  "real_phones": ("Real phones", "#eb6834"),
                  "real_phones_unbiased": ("Real phones, average bias removed", "#1baf7a")}
VENUES = [("concert", "Concert"), ("stadium", "Stadium"), ("f1", "F1 racetrack")]
LAYOUTS = [("random", "Random"), ("grid", "Grid"), ("clustered", "Clustered")]

INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def add_metrics(df):
    """RMSE splits into average bias and spread: RMSE^2 = bias^2 + spread^2.
    The spread is the error that is left once the map's average offset is removed."""
    df = df.copy()
    df["rmse_unbiased_db"] = np.sqrt(np.clip(df.rmse_db**2 - df.bias_db**2, 0, None))
    return df


def summarise(df, metric):
    g = df.groupby(["venue", "layout", "phone_count", "model"])[metric]
    return g.mean().rename("mean").to_frame().join(g.std().fillna(0).rename("sd")).reset_index()


def grid_figure(df, metric, ylabel, path):
    s = summarise(df, metric)
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.0), sharex=True)
    counts = sorted(df.phone_count.unique())
    for i, (layout, ltitle) in enumerate(LAYOUTS):
        for j, (venue, vtitle) in enumerate(VENUES):
            ax = axes[i, j]
            for model, (label, colour, ls, marker, filled) in STYLE.items():
                d = s[(s.venue == venue) & (s.layout == layout) & (s.model == model)].sort_values("phone_count")
                if d.empty:
                    continue
                ax.fill_between(d.phone_count, d["mean"] - d.sd, d["mean"] + d.sd,
                                color=colour, alpha=0.12, linewidth=0)
                ax.plot(d.phone_count, d["mean"], ls, color=colour, linewidth=1.6,
                        marker=marker, markersize=4.5, label=label,
                        markerfacecolor=colour if filled else "white", markeredgewidth=1.2)
            ax.set_xscale("log")
            ax.set_xticks(counts)
            ax.set_xticklabels([str(c) for c in counts])
            ax.minorticks_off()
            lows = s[(s.venue == venue) & (s.layout == layout)]
            bottom = min(0, (lows["mean"] - lows.sd).min() - 0.2)
            # keep one model's blow-up (e.g. invalid energy-kriging values) from
            # squashing every other line: cap the axis and name what is off scale
            limit = max(15.0, 3 * np.median(lows["mean"]))     # only real blow-ups
            normal = lows[lows["mean"] <= limit]
            top = (normal["mean"] + normal.sd).max() * 1.1
            off = sorted({STYLE[m][0] for m in lows[lows["mean"] > top].model if m in STYLE})
            if off:
                ax.set_ylim(bottom, top)
                ax.text(0.98, 0.97, "off scale:\n" + "\n".join(off), transform=ax.transAxes,
                        ha="right", va="top", fontsize=6.5, color=MUTED)
            else:
                ax.set_ylim(bottom=bottom)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(vtitle, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{ltitle}\n{ylabel}")
            if i == 2:
                ax.set_xlabel("Number of phones")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def latex_table(df, count, path):
    s = df[df.phone_count == count].groupby(["venue", "layout", "model"]).rmse_db
    mean, sd = s.mean().unstack("model"), s.std().unstack("model")
    models = [m for m in STYLE if m in mean.columns]
    lines = [r"\begin{tabular}{ll" + "c" * len(models) + "}", r"\toprule",
             "Venue & Layout & " + " & ".join(STYLE[m][0] for m in models) + r" \\", r"\midrule"]
    for venue, vtitle in VENUES:
        for k, (layout, ltitle) in enumerate(LAYOUTS):
            row = mean.loc[(venue, layout)]
            best = row[models].idxmin()
            cells = []
            for m in models:
                txt = f"{row[m]:.2f}"
                if layout != "grid":
                    txt += f" $\\pm$ {sd.loc[(venue, layout), m]:.2f}"
                cells.append(rf"\textbf{{{txt}}}" if m == best else txt)
            lines.append((vtitle if k == 0 else "") + f" & {ltitle} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if venue != VENUES[-1][0] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def readings_figure(perfect, real, path, table_path, count=50, layout="random"):
    """Bars per model: perfect phones, real phones, real phones with the bias removed."""
    models = [m for m in STYLE if m in perfect.model.unique() and m in real.model.unique()]
    pick = lambda d, metric: (d[(d.phone_count == count) & (d.layout == layout)]
                              .groupby(["venue", "model"])[metric].agg(["mean", "std"]))
    series = [("perfect", pick(perfect, "rmse_db")), ("real_phones", pick(real, "rmse_db")),
              ("real_phones_unbiased", pick(real, "rmse_unbiased_db"))]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True)
    width = 0.26
    for ax, (venue, vtitle) in zip(axes, VENUES):
        x = np.arange(len(models))
        for i, (key, table) in enumerate(series):
            label, colour = READINGS_STYLE[key]
            vals = [table.loc[(venue, m), "mean"] for m in models]
            errs = [table.loc[(venue, m), "std"] for m in models]
            ax.bar(x + (i - 1) * width, vals, width * 0.92, yerr=errs, color=colour,
                   label=label, error_kw=dict(ecolor=MUTED, elinewidth=0.8, capsize=1.5))
        ax.set_xticks(x)
        ax.set_xticklabels([STYLE[m][0] for m in models], rotation=60, ha="right", fontsize=7)
        ax.set_title(vtitle, fontsize=10)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"RMSE (dB), {count} {layout} phones")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)

    # the same numbers as a LaTeX table, plus the average bias with real phones
    bias = pick(real, "bias_db")
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Venue & Model & Perfect phones & Real phones & Real, bias removed & Real-phone bias \\",
             r"\midrule"]
    for venue, vtitle in VENUES:
        for k, m in enumerate(models):
            cells = [f"{t.loc[(venue, m), 'mean']:.2f} $\\pm$ {t.loc[(venue, m), 'std']:.2f}"
                     for _, t in series]
            cells.append(f"{bias.loc[(venue, m), 'mean']:+.2f}")
            lines.append((vtitle if k == 0 else "") + f" & {STYLE[m][0]} & "
                         + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if venue != VENUES[-1][0] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    table_path.write_text("\n".join(lines) + "\n")


def weights_figure(df, path):
    """Average blend weights the stacked ensemble chose, per venue and layout."""
    e = df[df.model == "ensemble"].copy()
    if e.empty:
        return
    parts = [("gp", "GP (tuned)", "#1baf7a"), ("idw", "IDW (dB)", "#2a78d6"),
             ("diffusion", "Diffusion LMS", "#c98500")]
    for key, _, _ in parts:
        e[key] = e.settings.str.extract(rf"w_{key}=([\d.]+)").astype(float)
    w = e.groupby(["venue", "layout"])[[k for k, _, _ in parts]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (venue, vtitle) in zip(axes, VENUES):
        layouts = [l for l, _ in LAYOUTS]
        bottom = np.zeros(len(layouts))
        for key, name, colour in parts:
            vals = np.array([w.loc[(venue, l), key] for l in layouts])
            ax.bar(range(len(layouts)), vals, 0.6, bottom=bottom, color=colour, label=name,
                   edgecolor="white", linewidth=1)
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v >= 0.12:
                    ax.text(i, b + v / 2, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="white")
            bottom += vals
        ax.set_xticks(range(len(layouts)))
        ax.set_xticklabels([t for _, t in LAYOUTS])
        ax.set_title(vtitle, fontsize=10)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Average blend weight")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readings", default="perfect")
    ap.add_argument("--models", nargs="+", default=None,
                    help="only plot these models (default: every model in the results)")
    ap.add_argument("--tag", default="",
                    help="write into figures/<tag>/ and tables/<tag>/ instead of the top level")
    args = ap.parse_args()

    def load(readings):
        d = add_metrics(pd.read_csv(RESULTS / f"model_comparison_{readings}.csv"))
        return d if args.models is None else d[d.model.isin(args.models)]

    df = load(args.readings)
    figs = RESULTS / "figures" / args.tag
    tables = RESULTS / "tables" / args.tag if args.tag else RESULTS
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    grid_figure(df, "rmse_db", "RMSE (dB)", figs / f"rmse_vs_phones_{args.readings}")
    grid_figure(df, "under_p95_db", "95th pct under-\nprediction (dB)",
                figs / f"under_p95_vs_phones_{args.readings}")

    if args.readings != "perfect":
        grid_figure(df, "rmse_unbiased_db", "RMSE, average\nbias removed (dB)",
                    figs / f"bias_removed_rmse_vs_phones_{args.readings}")
    perfect_csv = RESULTS / "model_comparison_perfect.csv"
    real_csv = RESULTS / "model_comparison_real_phones.csv"
    if perfect_csv.exists() and real_csv.exists():
        readings_figure(load("perfect"), load("real_phones"), figs / "perfect_vs_real_phones",
                        tables / "table_perfect_vs_real_phones.tex")
    weights_figure(df, figs / f"ensemble_weights_{args.readings}")

    metrics = ["rmse_db", "rmse_unbiased_db", "bias_db", "under_p95_db", "max_under_db",
               "within_3db", "n_invalid", "runtime_s"]
    summary = df.groupby(["venue", "layout", "phone_count", "model"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{a}" for m, a in summary.columns]
    summary.round(3).to_csv(tables / f"summary_{args.readings}.csv")
    latex_table(df, 50, tables / f"table_50_phones_{args.readings}.tex")
    print("Saved figures in", figs, "and tables in", tables)


if __name__ == "__main__":
    main()
