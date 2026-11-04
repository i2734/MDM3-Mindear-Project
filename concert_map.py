
"""
SoundTracker — O2 Academy Bristol-inspired concert simulation.

Simplified 2D synthetic sound field:
- Inverse-square attenuation
- Directional speakers
- Frequency-dependent wall absorption
- First- and second-order specular reflections
- Audience mask for external microphone placement

Geometry and source levels include modelling assumptions.
Sound levels are unweighted and are not measured venue data.
"""

from pathlib import Path
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from single_wall import (
    Wall,
    _intersect_parameters,
    material_properties
)


# ============================================================
# 1. SIMULATION PARAMETERS
# ============================================================

OUT = Path(__file__).resolve().parent

FREQUENCIES = (125, 250, 500, 1000, 2000, 4000)

# Assumed SPL at 1 m in EACH octave band, per equivalent source
SOURCE_DB_1M = 105.0

WIDTH = 24.0
LENGTH = 14.0

x = np.linspace(0, WIDTH, 193)
y = np.linspace(0, LENGTH, 113)

X, Y = np.meshgrid(x, y)

EPS = 1e-8


# ============================================================
# 2. SPEAKERS AND STAGE
# ============================================================

# x, y, aiming angle in degrees
SPEAKERS = [
    (6.5, 1.6, 82.0),
    (17.5, 1.6, 98.0)
]

# x, y, width, depth
STAGE = (6.0, 0.0, 12.0, 1.3)


# ============================================================
# 3. AUDIENCE GEOMETRY
# ============================================================

WALL_MARGIN = 0.5


# ============================================================
# 4. WALL GEOMETRY
# ============================================================

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


# ============================================================
# 5. AUDIENCE-ACCESSIBLE AREA
# ============================================================

def audience_mask(X, Y):
    """
    Define where audience members and phones can be located.

    Excludes:
    - The full strip behind the stage-front line
    - A 0.5 m margin in front of the stage
    - A 0.5 m margin from the venue walls

    The audience region is a modelling assumption.
    """

    sx, sy, sw, sh = STAGE

    # Audience begins in front of the stage across
    # the entire width, including the side regions.
    audience_front_y = sy + sh + WALL_MARGIN

    return (
        (X >= WALL_MARGIN)
        & (X <= WIDTH - WALL_MARGIN)
        & (Y >= audience_front_y)
        & (Y <= LENGTH - WALL_MARGIN)
    )


AUDIENCE_MASK = audience_mask(X, Y)


# ============================================================
# 6. ACOUSTIC FUNCTIONS
# ============================================================

def energy_from_db(level):
    return 10.0 ** (np.asarray(level) / 10.0)


def db_from_energy(energy):
    return 10.0 * np.log10(np.maximum(energy, 1e-30))


def directional_gain(sx, sy, angle_deg, px, py):
    """
    Illustrative directional speaker model.

    Not a measured L-Acoustics polar response.
    """

    relative = (
        np.arctan2(py - sy, px - sx)
        - np.deg2rad(angle_deg)
    )

    return (
        0.20
        + 0.80 * (0.5 * (1.0 + np.cos(relative))) ** 2
    )


def wall_object(wall):
    return Wall(wall[1], wall[2])


def material_for(wall, treated):
    """
    Treated wall sections use acoustic-panel absorption.
    Untreated sections use concrete absorption.
    """

    if treated and wall[3] == "panel":
        return "acoustic_panel"

    return "concrete"


def reflection_coefficient(wall, frequency, treated):
    """
    Approximate reflected energy fraction:

        rho = 1 - alpha

    Assumes negligible transmission through the backing.
    This is not a measured specular reflection coefficient.
    """

    _, alpha = material_properties(
        material_for(wall, treated),
        frequency
    )

    if alpha is None:
        return 0.0

    return max(0.0, 1.0 - alpha)


def mirror_point(px, py, wall):
    """Mirror a point across the infinite wall line."""

    ax, ay = wall[1]
    bx, by = wall[2]

    wx = bx - ax
    wy = by - ay

    q = (
        (px - ax) * wx
        + (py - ay) * wy
    ) / (wx**2 + wy**2)

    return (
        2 * (ax + q * wx) - px,
        2 * (ay + q * wy) - py
    )


def intersection(px, py, qx, qy, wall):
    """Intersection with a finite wall segment."""

    with np.errstate(divide="ignore", invalid="ignore"):

        t, u = _intersect_parameters(
            px, py, qx, qy, wall_object(wall)
        )

        hx = px + t * (qx - px)
        hy = py + t * (qy - py)

    valid = (
        np.isfinite(t)
        & np.isfinite(u)
        & (t > EPS)
        & (t < 1.0 - EPS)
        & (u >= -EPS)
        & (u <= 1.0 + EPS)
    )

    return hx, hy, valid


def unobstructed(px, py, qx, qy):
    """
    Check whether any wall crosses the interior
    of a sound propagation path.
    """

    shape = np.broadcast_shapes(
        np.shape(px),
        np.shape(py),
        np.shape(qx),
        np.shape(qy)
    )

    clear = np.ones(shape, dtype=bool)

    for wall in WALLS:

        with np.errstate(divide="ignore", invalid="ignore"):

            t, u = _intersect_parameters(
                px, py, qx, qy, wall_object(wall)
            )

        crosses = (
            np.isfinite(t)
            & np.isfinite(u)
            & (t > EPS)
            & (t < 1.0 - EPS)
            & (u >= -EPS)
            & (u <= 1.0 + EPS)
        )

        clear &= ~crosses

    return clear


def same_side(px, py, qx, qy, wall):
    """Check whether points are on the same side of a wall."""

    ax, ay = wall[1]
    bx, by = wall[2]

    wx = bx - ax
    wy = by - ay

    side_a = wx * (py - ay) - wy * (px - ax)
    side_b = wx * (qy - ay) - wy * (qx - ax)

    return side_a * side_b > EPS


# ============================================================
# 7. DIRECT SOUND
# ============================================================

def direct_energy(speaker, frequency, RX, RY):

    sx, sy, angle = speaker

    distance = np.maximum(
        np.hypot(RX - sx, RY - sy),
        1.0
    )

    intensity = (
        energy_from_db(SOURCE_DB_1M)
        * directional_gain(sx, sy, angle, RX, RY)
        / distance**2
    )

    return np.where(
        unobstructed(sx, sy, RX, RY),
        intensity,
        0.0
    )


# ============================================================
# 8. FIRST- AND SECOND-ORDER REFLECTIONS
# ============================================================

def reflected_energy(
    speaker,
    frequency,
    RX,
    RY,
    wall_sequence,
    treated=True
):
    """
    Image-source reflections from one or two walls.

    Validates finite-wall reflection points and checks
    propagation paths for wall intersections.
    """

    order = len(wall_sequence)

    if order not in (1, 2):
        raise ValueError(
            "Only first- and second-order reflections supported"
        )

    if (
        order == 2
        and wall_sequence[0] is wall_sequence[1]
    ):
        return np.zeros_like(RX, dtype=float)

    sx, sy, angle = speaker

    rho = 1.0

    images = [(sx, sy)]

    # Create successive image sources
    for wall in wall_sequence:

        rho *= reflection_coefficient(
            wall, frequency, treated
        )

        images.append(
            mirror_point(*images[-1], wall)
        )

    if rho <= 0:
        return np.zeros_like(RX, dtype=float)

    # Trace backwards from receiver
    bounce = [None] * order

    next_x, next_y = RX, RY

    valid = np.ones_like(RX, dtype=bool)

    for j in range(order - 1, -1, -1):

        hx, hy, hit_valid = intersection(
            *images[j + 1],
            next_x,
            next_y,
            wall_sequence[j]
        )

        bounce[j] = (hx, hy)

        valid &= hit_valid

        next_x, next_y = hx, hy

    # Actual propagation path
    points = [(sx, sy)] + bounce + [(RX, RY)]

    distance = np.zeros_like(RX, dtype=float)

    for j in range(len(points) - 1):

        px, py = points[j]
        qx, qy = points[j + 1]

        distance += np.hypot(
            qx - px,
            qy - py
        )

        valid &= unobstructed(
            px, py, qx, qy
        )

    # Validate reflection geometry
    for j, wall in enumerate(wall_sequence):

        valid &= same_side(
            *points[j],
            *points[j + 2],
            wall
        )

    # Speaker directionality towards first reflection
    first_x, first_y = bounce[0]

    gain = directional_gain(
        sx,
        sy,
        angle,
        first_x,
        first_y
    )

    intensity = (
        energy_from_db(SOURCE_DB_1M)
        * gain
        * rho
        / np.maximum(distance, 1.0)**2
    )

    return np.where(
        valid & np.isfinite(intensity),
        intensity,
        0.0
    )


# ============================================================
# 9. SIMULATE SOUND FIELD
# ============================================================

def simulate(treated=True, reflection_order=2):

    if reflection_order not in (0, 1, 2):
        raise ValueError(
            "reflection_order must be 0, 1 or 2"
        )

    band_maps = {}

    for frequency in FREQUENCIES:

        total = np.zeros_like(X)

        for speaker in SPEAKERS:

            # Direct sound
            total += direct_energy(
                speaker, frequency, X, Y
            )

            # First-order reflections
            if reflection_order >= 1:

                for wall in WALLS:

                    total += reflected_energy(
                        speaker,
                        frequency,
                        X,
                        Y,
                        (wall,),
                        treated
                    )

            # Second-order reflections
            if reflection_order >= 2:

                for wall1 in WALLS:

                    for wall2 in WALLS:

                        if wall1 is wall2:
                            continue

                        total += reflected_energy(
                            speaker,
                            frequency,
                            X,
                            Y,
                            (wall1, wall2),
                            treated
                        )

        band_maps[frequency] = db_from_energy(total)

    broadband = db_from_energy(
        sum(
            energy_from_db(field)
            for field in band_maps.values()
        )
    )

    return broadband, band_maps


# ============================================================
# 10. INTERPOLATION DATA INTERFACE
# ============================================================

@lru_cache(maxsize=4)
def _cached_field(treated=True, reflection_order=2):

    return simulate(
        treated=treated,
        reflection_order=reflection_order
    )[0]


def load_sound_map(treated=True, reflection_order=2):
    """
    Return:
        sound_maps   : (Ny, Nx) broadband sound field
        x            : x grid coordinates
        y            : y grid coordinates
        allowed_mask : audience-accessible positions
    """

    return (
        _cached_field(treated, reflection_order).copy(),
        x.copy(),
        y.copy(),
        AUDIENCE_MASK.copy()
    )


# ============================================================
# 11. VISUALISATION
# ============================================================

def draw_venue(ax, treated=True):
    """
    Draw venue geometry and permitted audience region.

    The grey fill corresponds to AUDIENCE_MASK.
    This is the same mask used for phone placement.
    """

    # Grey audience area
    audience_fill = np.ma.masked_where(
        ~AUDIENCE_MASK,
        np.ones_like(X)
    )

    ax.pcolormesh(
        x,
        y,
        audience_fill,
        cmap="Greys",
        vmin=0.0,
        vmax=2.0,
        alpha=0.35,
        shading="auto",
        zorder=2
    )

    # Light-grey audience boundary (not green)
    ax.contour(
        X,
        Y,
        AUDIENCE_MASK.astype(float),
        levels=[0.5],
        colors=["#D9DDE3"],
        linewidths=1.8,
        zorder=4
    )

    # Walls
    for _, start, end, kind in WALLS:

        color = (
            "cyan"
            if treated and kind == "panel"
            else "white"
        )

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=3,
            zorder=5
        )

    # Speakers
    for sx, sy, direction in SPEAKERS:

        ax.add_patch(
            Rectangle(
                (sx - 0.35, sy - 0.35),
                0.7,
                0.7,
                facecolor="darkslategrey",
                edgecolor="white",
                linewidth=1.4,
                zorder=7
            )
        )

        angle = np.deg2rad(direction)

        ax.annotate(
            "",
            xy=(
                sx + 1.7 * np.cos(angle),
                sy + 1.7 * np.sin(angle)
            ),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                color="red",
                lw=2.2
            ),
            zorder=8
        )

    # Stage
    ax.add_patch(
        Rectangle(
            STAGE[:2],
            STAGE[2],
            STAGE[3],
            facecolor="grey",
            alpha=0.8,
            zorder=6
        )
    )

    ax.text(
        12,
        0.65,
        "STAGE",
        color="white",
        ha="center",
        va="center",
        fontsize=8,
        zorder=10
    )

    ax.set(
        xlim=(-0.8, WIDTH + 0.8),
        ylim=(-0.8, LENGTH + 0.8),
        xlabel="x (m)",
        ylabel="y (m)"
    )

    ax.set_aspect("equal")


# ============================================================
# 12. OPTIONAL VENUE PLOTS AND ACOUSTIC SUMMARY
# ============================================================

def main():

    treated_db, band_maps = simulate(
        treated=True,
        reflection_order=2
    )

    untreated_db, _ = simulate(
        treated=False,
        reflection_order=2
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        constrained_layout=True
    )

    for ax, field, treated, title in zip(
        axes,
        (untreated_db, treated_db),
        (False, True),
        (
            "Untreated concrete",
            "Fiberglass-lined concrete"
        )
    ):

        im = ax.contourf(
            X,
            Y,
            field,
            levels=30,
            cmap="viridis"
        )

        draw_venue(ax, treated=treated)

        ax.set_title(title)

    fig.colorbar(
        im,
        ax=axes,
        label="Unweighted broadband SPL (dB)"
    )

    fig.savefig(
        OUT / "concert_venue_maps.png",
        dpi=180
    )

    plt.close(fig)

    audience = treated_db[AUDIENCE_MASK]

    print("Audience SPL (unweighted, dB):")

    print(
        f"Min {audience.min():.2f}, "
        f"Max {audience.max():.2f}, "
        f"Arithmetic mean {audience.mean():.2f}"
    )

    print(
        "Source level per octave band at 1 m:",
        SOURCE_DB_1M,
        "dB"
    )

    print(
        "Acoustic treatment: Fiberglass acoustic panels "
        "(1-inch rigid fiberglass board, OSHA Table V-1)"
    )

    print(
        "No microphone layout is defined here; "
        "use interpolation scripts."
    )


if __name__ == "__main__":
    main()
