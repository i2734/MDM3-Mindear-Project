
"""Compare random, grid and clustered microphone placements at different phone counts.

Loads the simulated sound map, generates 15 phone configurations and plots
their positions over the same sound field to compare spatial coverage.
"""

import inspect
import numpy as np
import matplotlib.pyplot as plt

# uncomment the appropriate line below to choose the venue sound map
import concert_map as sound_map
# import stadium_map as sound_map

from extract_sound_venues import prepare_interpolation_data


# load the simulated sound map and audience area
sound_maps, x, y, allowed_mask = sound_map.load_sound_map()

sound_maps = np.asarray(sound_maps)
x = np.asarray(x)
y = np.asarray(y)
allowed_mask = np.asarray(allowed_mask, dtype=bool)

# choose which time frame to display
frame = 80

if sound_maps.ndim == 3:
    frame = min(frame, len(sound_maps) - 1)
    plot_map = sound_maps[frame]
else:
    frame = 0
    plot_map = sound_maps


def draw_venue(ax):
    """Draw the venue layout, handling the different drawing functions used by the stadium and concert models."""
    # check whether the venue drawing function uses the treated argument
    if "treated" in inspect.signature(sound_map.draw_venue).parameters:
        sound_map.draw_venue(ax, treated=True)
    else:
        sound_map.draw_venue(ax)


# number of phones and layouts to compare
phone_counts = [10, 30, 50, 100, 250]
layouts = ["random", "grid", "clustered"]

# generate phone positions and extract the simulated sound measurements
prepared_data = prepare_interpolation_data(
    sound_maps=sound_maps,
    x=x,
    y=y,
    phone_counts=phone_counts,
    layouts=layouts,
    seed=42,
    allowed_mask=allowed_mask
)


# plot all 15 phone configurations
fig, axes = plt.subplots(len(phone_counts), len(layouts), figsize=(16, 19), constrained_layout=True)

# use the same sound level colour scale for every plot
vmin = float(np.nanmin(plot_map))
vmax = float(np.nanmax(plot_map))

for i, n in enumerate(phone_counts):
    for j, configuration in enumerate(layouts):

        ax = axes[i, j]
        positions = prepared_data[(n, configuration)]["phone_positions"]

        # plot the sound map for the selected time frame
        im = ax.pcolormesh(x, y, plot_map, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)

        draw_venue(ax)

        # show the microphone positions in red
        ax.scatter(
            positions[:, 0], positions[:, 1],
            c="red", s=24, edgecolors="white",
            linewidths=0.5, zorder=10
        )

        ax.set_title(f"{n} phones — {configuration}")

        if i != len(phone_counts) - 1:
            ax.set_xlabel("")

        if j != 0:
            ax.set_ylabel("")


# add one colour bar for all the plots
fig.colorbar(im, ax=axes, label="Sound level (dB)", shrink=0.6)
fig.suptitle(f"SoundTracker — Microphone Configurations (Frame {frame})", fontsize=17)

plt.savefig("all_phone_configurations.png", dpi=180, bbox_inches="tight")
plt.show()
