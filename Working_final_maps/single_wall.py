"""Calculate sound reaching receivers around a wall through transmission, diffraction and reflection.

Material properties vary by frequency. Sound contributions are combined in linear
energy units, rather than adding decibel levels directly.
"""

from dataclasses import dataclass
import numpy as np


# Transmission loss (TL) and absorption coefficients for each frequency.
# Values attributed to OSHA OTM Section III, Chapter 5, Tables V-1 to V-4.
MATERIALS = {
    "concrete": {
        "TL": {125: 38, 250: 36, 500: 40, 1000: 45, 2000: 50, 4000: 56},
        "alpha": {125: .10, 250: .05, 500: .06, 1000: .07, 2000: .09, 4000: .08}
    },
    "acoustic_panel": {
        "TL": None,
        "alpha": {125: .08, 250: .25, 500: .74, 1000: .95, 2000: .97, 4000: .99}
    }
}


@dataclass(frozen=True)
class Wall:
    """Store the endpoints of a wall segment and its material."""
    start: tuple
    end: tuple
    material: str = "concrete"


def material_properties(material, frequency):
    """Look up transmission loss in dB and absorption at the selected frequency."""
    props = MATERIALS[material]
    frequency = int(frequency)

    TL = None if props["TL"] is None else float(props["TL"][frequency])
    alpha = None if props["alpha"] is None else props["alpha"][frequency]

    return TL, alpha


def _cross(ax, ay, bx, by):
    """Calculate a 2D cross product to help find line intersections and sides."""
    return ax * by - ay * bx


def _intersect_parameters(sx, sy, X, Y, wall):
    """Find where a source-to-receiver line crosses the wall's line.

    t is the fraction along the sound path; u is the fraction along the wall.
    Both between 0 and 1 means the finite segments intersect.
    """
    ax, ay = wall.start
    bx, by = wall.end

    dx, dy = X - sx, Y - sy
    wx, wy = bx - ax, by - ay

    den = _cross(dx, dy, wx, wy)
    safe = np.where(np.abs(den) < 1e-12, np.nan, den)

    t = _cross(ax - sx, ay - sy, wx, wy) / safe
    u = _cross(ax - sx, ay - sy, dx, dy) / safe

    return t, u


def _blocked(sx, sy, X, Y, wall):
    """Mark receivers whose straight-line path from the source crosses the wall."""
    t, u = _intersect_parameters(sx, sy, X, Y, wall)
    return (t > 1e-9) & (t < 1 - 1e-9) & (u >= 0) & (u <= 1)


def _distance(sx, sy, X, Y, r_ref):
    """Find source-to-receiver distance, capped below at r_ref to avoid spikes."""
    return np.maximum(np.hypot(X - sx, Y - sy), r_ref)


def _knife_edge_loss(delta, wavelength):
    """Estimate diffraction loss in dB from extra travel distance around an edge."""
    v = np.sqrt(np.maximum(0.0, 2 * delta / wavelength))
    return 6.9 + 20 * np.log10(np.sqrt((v - 0.1)**2 + 1) + v - 0.1)


def _reflection_intensity(sx, sy, L_ref, X, Y, wall, r_ref, frequency=1000):
    """Estimate reflected sound using a mirror source and wall absorption.

    A reflection counts only if its path hits the finite wall and the source
    and receiver lie on the same side.
    """
    TL, alpha = material_properties(wall.material, frequency)

    # No reflection if absorption data is unavailable.
    if alpha is None:
        return np.zeros_like(np.asarray(X, dtype=float))

    # Estimate the reflected energy using the absorption coefficient.
    rho = max(0.0, 1.0 - alpha)

    if rho == 0:
        return np.zeros_like(np.asarray(X, dtype=float))

    ax, ay = wall.start
    bx, by = wall.end
    wx, wy = bx - ax, by - ay
    length2 = wx * wx + wy * wy

    # Mirror the sound source across the wall to calculate the reflection path.
    q = ((sx - ax) * wx + (sy - ay) * wy) / length2
    px, py = ax + q * wx, ay + q * wy
    mx, my = 2 * px - sx, 2 * py - sy

    t, u = _intersect_parameters(mx, my, X, Y, wall)

    source_side = _cross(wx, wy, sx - ax, sy - ay)
    receiver_side = _cross(wx, wy, X - ax, Y - ay)

    # Reflection is valid only if it hits the wall and returns to the source side.
    valid = ((t > 0) & (t < 1) & (u >= 0) & (u <= 1)
             & (source_side * receiver_side > 0))

    d = _distance(mx, my, X, Y, r_ref)

    return np.where(valid, rho * 10 ** (L_ref / 10) * (r_ref / d)**2, 0.0)


def source_intensity(sx, sy, L_ref, X, Y, walls=(), frequency=1000, r_ref=1.0):
    """Calculate sound reaching each receiver with or without one wall.

    Direct sound spreads with distance squared. Behind the wall, transmitted
    and diffracted sound replace the direct path; valid reflections are added.
    Only the first wall in 'walls' is used.
    """
    # Calculate the sound intensity without any walls.
    d = _distance(sx, sy, X, Y, r_ref)
    free = 10 ** (L_ref / 10) * (r_ref / d)**2

    if not walls:
        return free

    wall = walls[0]
    TL, _ = material_properties(wall.material, frequency)
    if TL is None:
        raise ValueError(
            f"No transmission loss data for '{wall.material}' at {frequency} Hz; "
            "this function needs TL to model sound passing through a wall."
        )

    # Check which receiver positions are blocked by the wall.
    blocked = _blocked(sx, sy, X, Y, wall)

    # Fraction of sound energy transmitted through the wall.
    tau = 10 ** (-TL / 10)

    ax, ay = wall.start
    bx, by = wall.end

    # Calculate the shortest path around either end of the wall.
    path_a = np.hypot(sx - ax, sy - ay) + np.hypot(X - ax, Y - ay)
    path_b = np.hypot(sx - bx, sy - by) + np.hypot(X - bx, Y - by)

    # Difference between the diffracted and direct path lengths.
    delta = np.maximum(np.minimum(path_a, path_b) - np.hypot(X - sx, Y - sy), 0.0)

    # Calculate the diffraction loss using the wavelength.
    loss = _knife_edge_loss(delta, 343.0 / frequency)

    # Combine transmitted and diffracted sound for positions behind the wall.
    screened = free * (tau + 10 ** (-loss / 10))
    direct_or_screened = np.where(blocked, screened, free)

    # Add the reflected sound energy.
    reflected = _reflection_intensity(sx, sy, L_ref, X, Y, wall, r_ref, frequency)

    return direct_or_screened + reflected
