"""F1 sound map using Austrian GP car positions."""

from pathlib import Path
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_closing, binary_fill_holes, binary_erosion, distance_transform_edt

BASE = Path(__file__).resolve().parent
CSV = BASE / 'synchronised_f1.csv'
LW_DB = 133.0
MIN_DISTANCE_M = 1.0
FPS = 30
FRAME_STEP = 1
GRID_NX, GRID_NY = 110, 85
MARGIN = 100
STAND_SETBACK_M = 22.0
STAND_DEPTH_M = 105.0

stand_corners = np.array([
    [-1100, 800],
    [650, 680],
    [720, -430],
    [-680, -520]
])

centre_corners = np.array([
    [-480, 365],
    [-380, 370],
    [-265, 315],
    [-310, 200],
    [-390, 135],
    [-465, 240]
])


def build_map():
    frame = pd.read_csv(CSV)
    drivers = sorted(frame['driver_number'].unique())
    times = np.sort(frame['time_s'].unique())
    px = frame.pivot(index='time_s', columns='driver_number', values='x_m').reindex(index=times, columns=drivers).to_numpy()
    py = frame.pivot(index='time_s', columns='driver_number', values='y_m').reindex(index=times, columns=drivers).to_numpy()

    xmin, xmax = frame.x_m.min() - 2*MARGIN, frame.x_m.max() + 2*MARGIN
    ymin, ymax = frame.y_m.min() - 2*MARGIN, frame.y_m.max() + 2*MARGIN
    x = np.linspace(xmin, xmax, GRID_NX, dtype=np.float32)
    y = np.linspace(ymin, ymax, GRID_NY, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    indices = np.arange(0, len(times), FRAME_STEP)
    sound_maps = np.empty((len(indices), GRID_NY, GRID_NX), dtype=np.float32)
    source_energy = np.float32(10 ** (LW_DB / 10) / (4 * np.pi))

    for k, ti in enumerate(indices):
        energy = np.zeros_like(X)
        for j in range(len(drivers)):
            d2 = (X - px[ti, j])**2 + (Y - py[ti, j])**2
            energy += source_energy / np.maximum(d2, MIN_DISTANCE_M**2)
        sound_maps[k] = 10 * np.log10(energy)

    track_rgba = plt.imread(BASE / 'f1_track_overlay.png')
    h, w = track_rgba.shape[:2]
    xx = np.linspace(xmin, xmax, w)
    yy = np.linspace(ymax, ymin, h)
    XX, YY = np.meshgrid(xx, yy)
    points = np.column_stack((XX.ravel(), YY.ravel()))

    track_mask = binary_closing(track_rgba[:, :, 3] > 0.05, iterations=2)
    track_inside = binary_fill_holes(track_mask)
    dx = (xmax - xmin) / (w - 1)
    dy = (ymax - ymin) / (h - 1)
    distance = distance_transform_edt(~track_inside, sampling=(dy, dx))

    within_boundary = MplPath(stand_corners).contains_points(points).reshape(h, w)
    outer_seating = (within_boundary & ~track_inside &
                     (distance >= STAND_SETBACK_M) &
                     (distance <= STAND_SETBACK_M + STAND_DEPTH_M))

    centre_mask = MplPath(centre_corners).contains_points(points).reshape(h, w)
    track_clearance = distance_transform_edt(~track_mask, sampling=(dy, dx))
    centre_seating = centre_mask & ~track_mask & (track_clearance >= 18)
    stands = outer_seating | centre_seating

    edge = stands & ~binary_erosion(stands, iterations=2)
    stands_rgba = np.zeros((h, w, 4), dtype=np.float32)
    stands_rgba[stands] = [0.98, 0.79, 0.19, 0.22]
    stands_rgba[edge] = [1.0, 0.93, 0.45, 0.90]

    # Sample the seating mask at the exact sound-map grid coordinates.
    ix = np.clip(np.rint((x - xmin) / (xmax - xmin) * (w - 1)).astype(int), 0, w - 1)
    iy = np.clip(np.rint((ymax - y) / (ymax - ymin) * (h - 1)).astype(int), 0, h - 1)
    allowed_mask = stands[np.ix_(iy, ix)]

    return sound_maps, x, y, allowed_mask, track_rgba, stands_rgba, drivers, times[indices], px[indices], py[indices]


_cache = None


def get_map():
    global _cache
    if _cache is None:
        _cache = build_map()
    return _cache


def load_sound_map():
    sound_maps, x, y, allowed_mask, *_ = get_map()
    return sound_maps, x, y, allowed_mask


def draw_venue(ax, show_cars=True, frame=80):
    _, x, y, _, track_rgba, stands_rgba, drivers, times, px, py = get_map()
    extent = [x[0], x[-1], y[0], y[-1]]

    stands_image = ax.imshow(stands_rgba, origin='upper', extent=extent, interpolation='bilinear', zorder=1)
    track_image = ax.imshow(track_rgba, origin='upper', extent=extent, interpolation='bilinear', alpha=0.5, zorder=2)

    if show_cars:
        frame = min(frame, len(times) - 1)
        colors = plt.get_cmap('tab20')
        for j, driver in enumerate(drivers):
            ax.plot(px[frame, j], py[frame, j], 's', color=colors(j % 20), markersize=5,
                    markeredgecolor='black', markeredgewidth=0.7, zorder=3)

    return stands_image, track_image

def animate():
    sound_maps, x, y, allowed_mask, _, _, drivers, times, px, py = get_map()
    # Use only spectator-area levels to set the colour scale
    spectator_levels = sound_maps[:, allowed_mask]
    vmin = float(np.min(spectator_levels))
    vmax = float(np.max(spectator_levels))

    extent = [x[0], x[-1], y[0], y[-1]]
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(sound_maps[0], origin='lower', extent=extent, cmap='inferno', vmin=vmin, vmax=vmax, interpolation='bilinear', animated=True, zorder=0)
    fig.colorbar(image, ax=ax, label='Sound Pressure Level (dB)')
    stands_image, track_image = draw_venue(ax, show_cars=False)
    track_image.set_animated(True)

    colors = plt.get_cmap('tab20')
    markers = []
    for j, driver in enumerate(drivers):
        dot, = ax.plot([], [], 's', color=colors(j % 20), markersize=5,
                       markeredgecolor='black', markeredgewidth=0.7,
                       zorder=3, animated=True)
        markers.append(dot)

    # Driver legend

    legend_handles = [
        Line2D(
            [0], [0], marker='s', linestyle='None',
            markerfacecolor=colors(j % 20), markeredgecolor='black',
            markeredgewidth=0.7, markersize=7, label=str(driver)
        )
        for j, driver in enumerate(drivers)
    ]

    legend = ax.legend(
        handles=legend_handles, title="Cars — driver numbers", loc='upper right',
        bbox_to_anchor=(0.993, 0.92), ncol=4,
        fontsize=8, markerscale=0.7, columnspacing=0.4,
        handletextpad=0.2, labelspacing=0.3, framealpha=0.85
    )

    ax.add_artist(legend)

    # Stands legend

    stands_handle = Patch(
        facecolor='#FACC30', edgecolor='#FFED73',
        linewidth=1, label='Stands / viewing areas'
    )

    stands_legend = ax.legend(
        handles=[stands_handle], loc='upper right',
        bbox_to_anchor=(0.993, 0.73), fontsize=8,
        framealpha=0.843, handlelength=2.0
)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title("F1 Sound Map — 17 Cars (Red Bull Ring-inspired)\n3-minute simulation")
    time_label = ax.text(0.98, 0.97, '', transform=ax.transAxes,
                         ha='right', va='top', fontsize=11,
                         bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'),
                         animated=True, zorder=5)
    fig.tight_layout()

    artists = [image, stands_image, track_image, *markers, time_label, legend, stands_legend]
    
    def update(k):
        image.set_data(sound_maps[k])
        for j in range(len(drivers)):
            markers[j].set_data([px[k, j]], [py[k, j]])
        time_label.set_text(f"Simulation Time: {times[k]:.1f} s / {times[-1]:.0f} s")
        return artists

    animation = FuncAnimation(fig, update, frames=len(times), interval=1000 / FPS,
                              blit=True, repeat=True, cache_frame_data=False)
    plt.show()
    return animation


if __name__ == '__main__':
    animate()
