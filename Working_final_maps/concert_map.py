"""Simulate the sound field in a concert venue with directional speakers and wall reflections.
Sound levels are simulated, unweighted and based on assumed source levels and venue geometry.
"""

from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from single_wall import Wall, _intersect_parameters, material_properties

FREQUENCIES = (125, 250, 500, 1000, 2000, 4000)
SOURCE_DB_1M = 105.0  # assumed SPL at 1m per speaker in each octave band
WIDTH, LENGTH = 24.0, 14.0
x = np.linspace(0, WIDTH, 193)
y = np.linspace(0, LENGTH, 113)
X, Y = np.meshgrid(x, y)
EPS = 1e-8

# speaker positions and directions (x, y, angle in degrees)
SPEAKERS = [(6.5, 1.6, 82.0), (17.5, 1.6, 98.0)]
STAGE = (6.0, 0.0, 12.0, 1.3)  # x, y, width, depth
WALL_MARGIN = 0.5

# wall sections, with panel sections treated only when treated=True
WALLS = [
    ("Front", (0., 0.), (24., 0.), "concrete"),
    ("Rear left", (0., 14.), (6., 14.), "concrete"),
    ("Rear treatment", (6., 14.), (18., 14.), "panel"),
    ("Rear right", (18., 14.), (24., 14.), "concrete"),
    ("Left lower", (0., 0.), (0., 4.), "concrete"),
    ("Left treatment", (0., 4.), (0., 11.), "panel"),
    ("Left upper", (0., 11.), (0., 14.), "concrete"),
    ("Right lower", (24., 0.), (24., 4.), "concrete"),
    ("Right treatment", (24., 4.), (24., 11.), "panel"),
    ("Right upper", (24., 11.), (24., 14.), "concrete")
]


def audience_mask(X, Y):
    """Identify the audience area where microphones can be placed."""
    # exclude the stage area and leave a 0.5m margin from the walls
    sx, sy, sw, sh = STAGE
    front_y = sy + sh + WALL_MARGIN
    return ((X >= WALL_MARGIN) & (X <= WIDTH - WALL_MARGIN)
            & (Y >= front_y) & (Y <= LENGTH - WALL_MARGIN))


AUDIENCE_MASK = audience_mask(X, Y)


def energy_from_db(level):
    """Convert decibel levels to linear energy values."""
    return 10.0 ** (np.asarray(level) / 10.0)


def db_from_energy(energy):
    """Convert linear energy values to decibels."""
    return 10.0 * np.log10(np.maximum(energy, 1e-30))


def directional_gain(sx, sy, angle_deg, px, py):
    """Estimate how speaker direction affects sound at a position."""
    # assumed speaker directionality, not a measured speaker response
    relative = np.arctan2(py - sy, px - sx) - np.deg2rad(angle_deg)
    return 0.20 + 0.80 * (0.5 * (1.0 + np.cos(relative))) ** 2


def wall_object(wall):
    """Convert a wall definition into a Wall object."""
    return Wall(wall[1], wall[2])


def material_for(wall, treated):
    """Select the wall material based on whether treatment is applied."""
    return "acoustic_panel" if treated and wall[3] == "panel" else "concrete"


def reflection_coefficient(wall, frequency, treated):
    """Estimate the fraction of sound energy reflected by a wall."""
    # reflected energy estimated as 1 - absorption, assuming negligible transmission
    _, alpha = material_properties(material_for(wall, treated), frequency)
    return 0.0 if alpha is None else max(0.0, 1.0 - alpha)


def mirror_point(px, py, wall):
    """Reflect a point across a wall for the image-source method."""
    # mirror the source across the wall to calculate a reflection
    ax, ay = wall[1]
    bx, by = wall[2]
    wx, wy = bx - ax, by - ay
    q = ((px - ax) * wx + (py - ay) * wy) / (wx**2 + wy**2)
    return 2 * (ax + q * wx) - px, 2 * (ay + q * wy) - py


def intersection(px, py, qx, qy, wall):
    """Find where a sound path intersects a finite wall."""
    # find where a sound path meets the finite wall
    with np.errstate(divide="ignore", invalid="ignore"):
        t, u = _intersect_parameters(px, py, qx, qy, wall_object(wall))
        hx, hy = px + t * (qx - px), py + t * (qy - py)
    valid = (np.isfinite(t) & np.isfinite(u) & (t > EPS) & (t < 1.0 - EPS)
             & (u >= -EPS) & (u <= 1.0 + EPS))
    return hx, hy, valid


def unobstructed(px, py, qx, qy):
    """Check whether a sound path crosses any venue wall."""
    # check whether any wall blocks the sound path
    shape = np.broadcast_shapes(np.shape(px), np.shape(py), np.shape(qx), np.shape(qy))
    clear = np.ones(shape, dtype=bool)
    for wall in WALLS:
        with np.errstate(divide="ignore", invalid="ignore"):
            t, u = _intersect_parameters(px, py, qx, qy, wall_object(wall))
        crosses = (np.isfinite(t) & np.isfinite(u) & (t > EPS) & (t < 1.0 - EPS)
                   & (u >= -EPS) & (u <= 1.0 + EPS))
        clear &= ~crosses
    return clear


def same_side(px, py, qx, qy, wall):
    """Check whether two points lie on the same side of a wall."""
    ax, ay = wall[1]
    bx, by = wall[2]
    wx, wy = bx - ax, by - ay
    side_a = wx * (py - ay) - wy * (px - ax)
    side_b = wx * (qy - ay) - wy * (qx - ax)
    return side_a * side_b > EPS


def direct_energy(speaker, frequency, RX, RY):
    """Calculate direct sound energy using distance and speaker direction."""
    # direct sound, reducing with distance and speaker direction
    sx, sy, angle = speaker
    distance = np.maximum(np.hypot(RX - sx, RY - sy), 1.0)
    intensity = energy_from_db(SOURCE_DB_1M) * directional_gain(sx, sy, angle, RX, RY) / distance**2
    return np.where(unobstructed(sx, sy, RX, RY), intensity, 0.0)


def reflected_energy(speaker, frequency, RX, RY, wall_sequence, treated=True):
    """Calculate sound energy reflected from one or two walls."""
    # calculate sound reflected from one or two walls using image sources
    order = len(wall_sequence)
    if order == 2 and wall_sequence[0] is wall_sequence[1]:
        return np.zeros_like(RX, dtype=float)

    sx, sy, angle = speaker
    rho = 1.0
    images = [(sx, sy)]

    # mirror the source across each wall and multiply the reflection coefficients
    for wall in wall_sequence:
        rho *= reflection_coefficient(wall, frequency, treated)
        images.append(mirror_point(*images[-1], wall))

    if rho <= 0:
        return np.zeros_like(RX, dtype=float)

    # trace back from each receiver to find the reflection points
    bounce = [None] * order
    next_x, next_y = RX, RY
    valid = np.ones_like(RX, dtype=bool)
    for j in range(order - 1, -1, -1):
        hx, hy, hit_valid = intersection(*images[j + 1], next_x, next_y, wall_sequence[j])
        bounce[j] = (hx, hy)
        valid &= hit_valid
        next_x, next_y = hx, hy

    # add up the path lengths and check none of the paths cross another wall
    points = [(sx, sy)] + bounce + [(RX, RY)]
    distance = np.zeros_like(RX, dtype=float)
    for j in range(len(points) - 1):
        px, py = points[j]
        qx, qy = points[j + 1]
        distance += np.hypot(qx - px, qy - py)
        valid &= unobstructed(px, py, qx, qy)

    for j, wall in enumerate(wall_sequence):
        valid &= same_side(*points[j], *points[j + 2], wall)

    # speaker direction towards the first reflection
    first_x, first_y = bounce[0]
    gain = directional_gain(sx, sy, angle, first_x, first_y)
    intensity = energy_from_db(SOURCE_DB_1M) * gain * rho / np.maximum(distance, 1.0)**2
    return np.where(valid & np.isfinite(intensity), intensity, 0.0)


def simulate(treated=True, reflection_order=2):
    """Calculate the broadband and octave-band sound maps."""
    band_maps = {}
    for frequency in FREQUENCIES:
        total = np.zeros_like(X)
        for speaker in SPEAKERS:
            total += direct_energy(speaker, frequency, X, Y)

            # first reflections from each wall
            if reflection_order >= 1:
                for wall in WALLS:
                    total += reflected_energy(speaker, frequency, X, Y, (wall,), treated)

            # second reflections from pairs of different walls
            if reflection_order >= 2:
                for wall1 in WALLS:
                    for wall2 in WALLS:
                        if wall1 is wall2:
                            continue
                        total += reflected_energy(speaker, frequency, X, Y, (wall1, wall2), treated)

        band_maps[frequency] = db_from_energy(total)

    # sum energy across octave bands before converting back to dB
    broadband = db_from_energy(sum(energy_from_db(field) for field in band_maps.values()))
    return broadband, band_maps


@lru_cache(maxsize=4)
def _cached_field(treated=True, reflection_order=2):
    """Cache the broadband sound map to avoid repeated simulations."""
    return simulate(treated=treated, reflection_order=reflection_order)[0]


def load_sound_map(treated=True, reflection_order=2):
    """Return the sound map, coordinates and audience mask for interpolation."""
    # return the simulated sound field and audience area for microphone placement
    return _cached_field(treated, reflection_order).copy(), x.copy(), y.copy(), AUDIENCE_MASK.copy()


def draw_venue(ax, treated=True):
    """Draw the venue walls, audience outline, stage and labelled speakers."""
    # acoustic panel sections are cyan and concrete walls are white
    for _, start, end, kind in WALLS:
        color = "cyan" if treated and kind == "panel" else "white"
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=3, zorder=5)

    # outline the audience area without covering the sound map
    front_y = STAGE[1] + STAGE[3] + WALL_MARGIN
    ax.add_patch(Rectangle((WALL_MARGIN, front_y), WIDTH - 2 * WALL_MARGIN,
                           LENGTH - WALL_MARGIN - front_y, fill=False,
                           edgecolor="white", linestyle="--", linewidth=1.2, zorder=6))

    # show speaker positions, directions and short labels
    for i, (sx, sy, direction) in enumerate(SPEAKERS, start=1):
        ax.add_patch(Rectangle((sx - 0.35, sy - 0.35), 0.7, 0.7,
                               facecolor="black", edgecolor="white", linewidth=1.4, zorder=7))
        angle = np.deg2rad(direction)
        ax.annotate("", xy=(sx + 1.7 * np.cos(angle), sy + 1.7 * np.sin(angle)),
                    xytext=(sx, sy), arrowprops=dict(arrowstyle="-|>", color="red", lw=2.2), zorder=8)
        ax.text(sx, sy - 0.55, f"Speaker{i}", color="white", ha="center", va="top",
                fontsize=10, fontweight="bold", zorder=10)

    ax.add_patch(Rectangle(STAGE[:2], STAGE[2], STAGE[3], facecolor="darkslategrey", alpha=1, zorder=6))
    ax.text(12, 0.65, "STAGE", color="white", ha="center", va="center", fontsize=10, fontweight="bold",zorder=10)
    ax.set(xlim=(-0.8, WIDTH + 0.8), ylim=(-0.8, LENGTH + 0.8), xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal")


def main():
    """Plot the sound map for the acoustically treated concert venue."""
    sound_db, _ = simulate(treated=True, reflection_order=2)
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    im = ax.pcolormesh(x, y, sound_db, shading="auto", cmap="viridis")
    draw_venue(ax, treated=True)
    ax.set_title("Concert Sound Map (O2 Academy Bristol-inspired)")
    fig.colorbar(im, ax=ax, label="Sound Pressure Level (dB)")
    plt.show()


if __name__ == "__main__":
    main()
