"""Make the report figures and summary table from compare_models.py output.

    python3 analysis/plot_comparison.py                       # uses model_comparison_perfect.csv
    python3 analysis/plot_comparison.py --readings noisy

Writes to analysis/results/:
    figures/rmse_vs_phones_<readings>.pdf/.png         RMSE vs phone count
    figures/under_p95_vs_phones_<readings>.pdf/.png    95th-percentile under-prediction
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
    "idw":       ("IDW",                 "#2a78d6", "-",  "o", True),
    "kriging":   ("Ordinary kriging",    "#eb6834", "-",  "s", True),
    "gp_tuned":  ("GP (tuned length scale)", "#1baf7a", "-",  "^", True),
    "gp":        ("GP (fixed 15 m)",     "#1baf7a", "--", "^", False),
    "diffusion": ("Diffusion LMS",       "#c98500", "-",  "D", True),
}
VENUES = [("concert", "Concert"), ("stadium", "Stadium"), ("f1", "F1 racetrack")]
LAYOUTS = [("random", "Random"), ("grid", "Grid"), ("clustered", "Clustered")]

INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


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
            ax.set_ylim(bottom=min(0, (lows["mean"] - lows.sd).min() - 0.2))
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
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readings", default="perfect")
    args = ap.parse_args()
    df = pd.read_csv(RESULTS / f"model_comparison_{args.readings}.csv")
    figs = RESULTS / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    grid_figure(df, "rmse_db", "RMSE (dB)", figs / f"rmse_vs_phones_{args.readings}")
    grid_figure(df, "under_p95_db", "95th pct under-\nprediction (dB)",
                figs / f"under_p95_vs_phones_{args.readings}")

    metrics = ["rmse_db", "bias_db", "under_p95_db", "max_under_db", "within_3db", "runtime_s"]
    summary = df.groupby(["venue", "layout", "phone_count", "model"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{a}" for m, a in summary.columns]
    summary.round(3).to_csv(RESULTS / f"summary_{args.readings}.csv")
    latex_table(df, 50, RESULTS / f"table_50_phones_{args.readings}.tex")
    print("Saved figures and tables in", RESULTS)


if __name__ == "__main__":
    main()
