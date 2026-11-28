
"""Three ground-truth maps with individual colour bars and legends."""

import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import concert_map
import stadium_map
import race_track_map

OUTPUT = "ground_truth_combined.png"
STADIUM_FRAME = 60
F1_FRAME = 80

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 19,
    "axes.titlesize": 25,
    "axes.labelsize": 22,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 19,
})


def load_venue(module, requested_frame):
    maps, x, y, _ = module.load_sound_map()
    maps = np.asarray(maps)

    frame = (
        min(requested_frame, len(maps) - 1)
        if maps.ndim == 3 else 0
    )

    field = maps[frame] if maps.ndim == 3 else maps

    return {
        "module": module,
        "field": field,
        "x": np.asarray(x),
        "y": np.asarray(y),
        "frame": frame,
    }


venues = [
    load_venue(concert_map, 0),
    load_venue(stadium_map, STADIUM_FRAME),
    load_venue(race_track_map, F1_FRAME),
]

titles = ["(a) Concert", "(b) Stadium", "(c) Formula 1"]


def draw_venue(ax, venue):
    module = venue["module"]
    params = inspect.signature(module.draw_venue).parameters

    if module.__name__ == "concert_map":
        if "treated" in params:
            module.draw_venue(ax, treated=True)
        else:
            module.draw_venue(ax)

        # Adjust arrow length after drawing
        for txt in ax.texts:
            if hasattr(txt, "arrow_patch") and txt.arrow_patch is not None:
                x_start, y_start = txt.get_position()
                x_end, y_end = txt.xy

                scale = 1.5

                txt.xy = (
                    x_start + scale * (x_end - x_start),
                    y_start + scale * (y_end - y_start),
                )

    elif module.__name__ == "race_track_map":
        kwargs = {}

        if "show_cars" in params:
            kwargs["show_cars"] = True

        if "frame" in params:
            kwargs["frame"] = venue["frame"]

        module.draw_venue(ax, **kwargs)

    else:
        module.draw_venue(ax)


def get_legend(module):
    if module.__name__ == "concert_map":
        return [
            Line2D(
                [0], [0],
                color="#00FFFF",
                lw=3,
                label="Acoustic panels",
            ),
            Line2D(
                [0], [0],
                color="#A9A9A9",
                lw=2,
                label="Concrete walls",
            ),
        ]

    if module.__name__ == "stadium_map":
        return [
            Line2D(
                [0], [0],
                color="#A9A9A9",
                lw=2,
                label="Concrete walls",
            ),
        ]

    return [
        Line2D(
            [0], [0],
            color="#A9A9A9",
            linewidth=3,
            label="Race track",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markerfacecolor="#777777",
            markeredgecolor="black",
            markersize=6,
            label="Race cars (different colours)",
        ),
        Patch(
            facecolor="#705039",
            edgecolor="#B69A53",
            linewidth=0.8,
            label="Viewing areas",
        ),
    ]


# Each venue has a dedicated vertical stack:
# title/map, colour bar, and legend.
fig = plt.figure(figsize=(15, 7))

outer = fig.add_gridspec(
    1, 3,
    left=0.065,
    right=0.985,
    top=0.93,
    bottom=0.06,
    wspace=0.38,
)

for i, (venue, title) in enumerate(zip(venues, titles)):

    inner = outer[0, i].subgridspec(
        3, 1,
        height_ratios=[1, 0.065, 0.20],
        hspace=0.65,
    )

    ax = fig.add_subplot(inner[0])
    cax = fig.add_subplot(inner[1])
    lax = fig.add_subplot(inner[2])
    lax.axis("off")

    field = venue["field"]

    im = ax.pcolormesh(
        venue["x"],
        venue["y"],
        field,
        cmap="inferno",
        shading="auto",
        vmin=float(np.nanmin(field)),
        vmax=float(np.nanmax(field)),
        rasterized=True,
    )

    draw_venue(ax, venue)

    ax.set_title(title, fontsize=25, pad=10)
    ax.set_xlabel("x (m)", fontsize=22)
    ax.set_ylabel("y (m)", fontsize=22)
    ax.tick_params(labelsize=19, pad=3)
    ax.set_aspect("equal", adjustable="box")

    # Reduce tick density to prevent overlapping labels
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # Adjust internal map labels
    for txt in ax.texts:
        label = txt.get_text().strip().lower()

        if "speaker" in label:
            txt.set_fontsize(11)
            x_pos, _ = txt.get_position()

            if "speaker 1" in label:
                txt.set_position((x_pos - 1.5, 0))
            elif "speaker 2" in label:
                txt.set_position((x_pos + 1.5, 0))

            txt.set_verticalalignment("bottom")
            txt.set_bbox(dict(
                facecolor="black",
                alpha=0.35,
                edgecolor="none",
                pad=1.5,
            ))

        elif label == "stage":
            txt.set_fontsize(11)

        elif label in ["atyeo", "lansdown", "dolman", "south"]:
            txt.set_fontsize(12)
            txt.set_bbox(dict(
                facecolor="black",
                alpha=0.65,
                edgecolor="none",
                pad=1.5,
            ))

    cb = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
    )

    cb.locator = MaxNLocator(nbins=4)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=19, pad=3)
    cb.set_label("SPL (dB)", fontsize=22, labelpad=4)

    handles = get_legend(venue["module"])

    lax.legend(
        handles=handles,
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=19,
        handlelength=1.6,
        columnspacing=1.0,
        handletextpad=0.4,
        labelspacing=0.3,
    )

fig.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.06,
)

plt.show()
print(f"Saved: {OUTPUT}")
