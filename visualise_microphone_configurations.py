
"""Compare random, grid and clustered microphone placements."""

import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# Select venue
# import concert_map as sound_map
# import stadium_map as sound_map
import race_track_map as sound_map

from extract_sound_venues import prepare_interpolation_data


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 23,
    "axes.titlesize": 28,
    "axes.labelsize": 28,
    "xtick.labelsize": 23,
    "ytick.labelsize": 23,
    "figure.titlesize": 35,
    "legend.fontsize": 29,
})

PHONE_COUNTS = [10, 30, 50, 100, 250]
LAYOUTS = ["random", "grid", "clustered"]

FRAME = 80
SEED = 42
MICROPHONE_COLOR = "#39FF14"


sound_maps, x, y, allowed_mask = sound_map.load_sound_map()

sound_maps = np.asarray(sound_maps)
x = np.asarray(x)
y = np.asarray(y)
allowed_mask = np.asarray(allowed_mask, dtype=bool)

if sound_maps.ndim == 3:
    frame = min(FRAME, len(sound_maps) - 1)
    plot_map = sound_maps[frame]
else:
    frame = 0
    plot_map = sound_maps


def draw_venue(ax):
    params = inspect.signature(sound_map.draw_venue).parameters

    if sound_map.__name__ == "concert_map":
        if "treated" in params:
            sound_map.draw_venue(ax, treated=True)
        else:
            sound_map.draw_venue(ax)

        for txt in ax.texts:
            if hasattr(txt, "arrow_patch") and txt.arrow_patch is not None:
                x_start, y_start = txt.get_position()
                x_end, y_end = txt.xy

                scale = 1.5

                txt.xy = (
                    x_start + scale * (x_end - x_start),
                    y_start + scale * (y_end - y_start),
                )

    elif sound_map.__name__ == "race_track_map":
        kwargs = {}

        if "show_cars" in params:
            kwargs["show_cars"] = True

        if "frame" in params:
            kwargs["frame"] = frame

        sound_map.draw_venue(ax, **kwargs)

    else:
        sound_map.draw_venue(ax)

    for txt in ax.texts:
        label = txt.get_text().strip().lower()

        if "speaker" in label or label in [
            "stage", "atyeo", "lansdown", "dolman", "south"
        ]:
            txt.set_visible(False)

    for txt in ax.texts:
        label = txt.get_text().strip().lower()

        if "speaker" in label:
            txt.set_fontsize(17)
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
            txt.set_fontsize(17)

        elif label in ["atyeo", "lansdown", "dolman", "south"]:

            if label in ["lansdown", "dolman"]:
                txt.set_fontsize(13)
            else:
                txt.set_fontsize(16)

            txt.set_bbox(dict(
                facecolor="black",
                alpha=0.65,
                edgecolor="none",
                pad=1.0,
            ))


prepared_data = prepare_interpolation_data(
    sound_maps=sound_maps,
    x=x,
    y=y,
    phone_counts=PHONE_COUNTS,
    layouts=LAYOUTS,
    seed=SEED,
    allowed_mask=allowed_mask,
)


if sound_map.__name__ == "concert_map":
    venue_name = "Concert"
    time_text = "Static sound field"

elif sound_map.__name__ == "stadium_map":
    venue_name = "Stadium"
    time_text = (
        f"Simulation time: "
        f"{frame * sound_map.TIME_STEP_SECONDS:.0f} s / "
        f"{sound_map.DURATION_SECONDS:.0f} s"
    )

else:
    venue_name = "F1"
    times = sound_map.get_map()[7]
    time_text = (
        f"Simulation time: "
        f"{times[frame]:.1f} s / {times[-1]:.0f} s"
    )


fig = plt.figure(figsize=(23, 30))

gs = fig.add_gridspec(
    nrows=6,
    ncols=4,
    left=0.075,
    right=0.92,
    top=0.91,
    bottom=0.045,
    width_ratios=[1, 1, 1, 0.10],
    height_ratios=[1, 1, 1, 1, 1, 0.50],
    hspace=0.55,
    wspace=0.12,
)

axes = np.array([
    [fig.add_subplot(gs[i, j]) for j in range(3)]
    for i in range(5)
])

vmin = float(np.nanmin(plot_map))
vmax = float(np.nanmax(plot_map))


for i, n in enumerate(PHONE_COUNTS):

    for j, configuration in enumerate(LAYOUTS):

        ax = axes[i, j]

        positions = prepared_data[
            (n, configuration)
        ]["phone_positions"]

        im = ax.pcolormesh(
            x,
            y,
            plot_map,
            shading="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

        draw_venue(ax)

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=MICROPHONE_COLOR,
            s=45,
            edgecolors="black",
            linewidths=1,
            zorder=10,
        )

        ax.set_title(
            f"{n} phones — {configuration}",
            fontsize=28,
            pad=10,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=23, pad=3)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        if i == len(PHONE_COUNTS) - 1:
            ax.set_xlabel("x (m)", fontsize=28)
        else:
            ax.set_xlabel("")

        if j == 0:
            ax.set_ylabel("y (m)", fontsize=28)
        else:
            ax.set_ylabel("")


cbar_ax = fig.add_subplot(gs[:5, 3])
cbar_ax.set_box_aspect(None)

cbar = fig.colorbar(im, cax=cbar_ax)

cbar.locator = MaxNLocator(nbins=8)
cbar.update_ticks()
cbar.ax.tick_params(labelsize=23, pad=3)

cbar.set_label(
    "SPL (dB)",
    fontsize=28,
    labelpad=15,
)


microphone_handle = Line2D(
    [0], [0],
    marker="o",
    linestyle="None",
    markerfacecolor=MICROPHONE_COLOR,
    markeredgecolor="black",
    markersize=12,
    label="Microphone positions",
)

if venue_name == "Concert":

    main_handles = [
        microphone_handle,
        Line2D(
            [0], [0],
            color="#00FFFF",
            linewidth=4,
            label="Acoustic panels",
        ),
        Line2D(
            [0], [0],
            color="#A9A9A9",
            linewidth=3,
            label="Concrete walls",
        ),
    ]

elif venue_name == "Stadium":

    main_handles = [
        microphone_handle,
        Line2D(
            [0], [0],
            color="#A9A9A9",
            linewidth=3,
            label="Concrete walls",
        ),
    ]

else:

    main_handles = [
        microphone_handle,
        Line2D(
            [0], [0],
            color="#A9A9A9",
            linewidth=3,
            label="Race track",
        ),
        Patch(
            facecolor="#705039",
            edgecolor="#B69A53",
            linewidth=0.8,
            label="Viewing areas",
        ),
    ]


if venue_name == "F1":

    legend_gs = gs[5, :3].subgridspec(
        1, 2,
        width_ratios=[1.6, 3.4],
        wspace=0.08,
    )

    left_ax = fig.add_subplot(legend_gs[0, 0])
    left_ax.axis("off")

    left_ax.legend(
        handles=main_handles,
        loc="center left",
        ncol=1,
        frameon=False,
        fontsize=29,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.4,
    )

    right_ax = fig.add_subplot(legend_gs[0, 1])
    right_ax.axis("off")

    colors = plt.get_cmap("tab20")
    drivers = sound_map.get_map()[6]

    car_handles = [
        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markerfacecolor=colors(j % 20),
            markeredgecolor="black",
            markersize=10,
            label=f"Car {driver}",
        )
        for j, driver in enumerate(drivers)
    ]

    right_ax.legend(
        handles=car_handles,
        loc="center",
        ncol=9,
        frameon=False,
        fontsize=20,
        title="Race cars — driver numbers",
        title_fontsize=25,
        columnspacing=0.7,
        handletextpad=0.3,
        labelspacing=0.4,
        handlelength=0.8,
        borderaxespad=0,
    )

else:

    legend_ax = fig.add_subplot(gs[5, :3])
    legend_ax.axis("off")

    legend_ax.legend(
        handles=main_handles,
        loc="center",
        ncol=len(main_handles),
        frameon=False,
        fontsize=29,
        handlelength=1.6,
        columnspacing=1.5,
        handletextpad=0.5,
        labelspacing=0.4,
    )


fig.suptitle(
    f"{venue_name} — Microphone Configurations\n{time_text}",
    fontsize=35,
    y=0.985,
)


output_filename = (
    f"{venue_name.lower()}_microphone_configurations.png"
)

fig.savefig(
    output_filename,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.06,
)

plt.show()
print(f"Saved: {output_filename}")
