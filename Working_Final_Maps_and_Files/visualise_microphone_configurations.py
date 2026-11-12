
"""Compare random, grid and clustered microphone placements at different phone counts.

Loads the simulated sound map, generates 15 phone configurations and plots
their positions over the same sound field to compare spatial coverage.
"""

import inspect
import numpy as np
import matplotlib.pyplot as plt

# uncomment the appropriate line below to choose the venue sound map
# import concert_map as sound_map
# import stadium_map as sound_map
import race_track_map as sound_map

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
    if "treated" in inspect.signature(sound_map.draw_venue).parameters:
        sound_map.draw_venue(ax, treated=True)
    elif sound_map.__name__ == "race_track_map":
        sound_map.draw_venue(ax, show_cars=True, frame=frame)
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
    allowed_mask=allowed_mask)


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
        im = ax.pcolormesh(x, y, plot_map, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)

        draw_venue(ax)

        # show the microphone positions in neon green
        ax.scatter(positions[:, 0], positions[:, 1],c="#39FF14", s=45,edgecolors="black", linewidths=1, zorder=10, label="Microphones")

        ax.set_title(f"{n} phones — {configuration}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")

        if i != len(phone_counts) - 1:
            ax.set_xlabel("")

        if j != 0:
            ax.set_ylabel("")


# add one colour bar for all the plots
fig.colorbar(im, ax=axes, label="Sound Pressure Level (dB)", shrink=0.6)    
if sound_map.__name__ == "concert_map":
    venue_name = "Concert"
    time_text = "Static sound field"
elif sound_map.__name__ == "stadium_map":
    venue_name = "Stadium"
    time_text = f"Simulation time: {frame * sound_map.TIME_STEP_SECONDS:.0f} s / {sound_map.DURATION_SECONDS:.0f} s"
else:
    venue_name = "F1"
    times = sound_map.get_map()[7]
    time_text = f"Simulation time: {times[frame]:.1f} s / {times[-1]:.0f} s"

fig.suptitle(f"{venue_name} — Microphone Configurations\n{time_text}", fontsize=17)

fig.legend(
    handles=[plt.Line2D([0], [0], marker='o', linestyle='None',
                        markerfacecolor='#39FF14', markeredgecolor='black',
                        label='Microphone positions')],
    loc='lower center',
    bbox_to_anchor=(0.5, 0.005),
    frameon=False
)

plt.savefig(f"{venue_name.lower()}_microphone_configurations.png", dpi=180, bbox_inches="tight")
plt.show()
