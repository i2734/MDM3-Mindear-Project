"""Simplified 2D finite-wall sound model, evaluated per octave band.

Incoherent energy summation; endpoint diffraction approximation; at most one
specular reflection per wall. Material data describe specific constructions,
not universal material constants. Missing absorption data omit reflections.
OSHA OTM Section III Chapter 5: concrete alpha Table V-2, concrete TL Table V-4,
steel TL Table V-3, fiberglass alpha Table V-1.
Fiberglass panel TL is unavailable for the SAME construction, so None.
"""
from dataclasses import dataclass
import numpy as np

MATERIALS = {
    "concrete": {
        "TL": {125: 38, 250: 36, 500: 40, 1000: 45, 2000: 50, 4000: 56},
        "alpha": {125: .10, 250: .05, 500: .06, 1000: .07, 2000: .09, 4000: .08},
    },
    "acoustic_panel": {
        "TL": None,
        "alpha": {125: .08, 250: .25, 500: .74, 1000: .95, 2000: .97, 4000: .99},
    },
}

@dataclass(frozen=True)
class Wall:
    start: tuple
    end: tuple
    material: str = "concrete"


def material_properties(material, frequency):
    """Return (TL or None, alpha or None) for an octave-band frequency."""
    if material not in MATERIALS:
        raise ValueError(f"Unknown material: {material}")
    frequency = int(frequency)
    props = MATERIALS[material]
    if frequency not in (125, 250, 500, 1000, 2000, 4000):
        raise ValueError(f"Unsupported frequency: {frequency} Hz")
    TL = None if props["TL"] is None else props["TL"][frequency]
    alpha = props["alpha"]
    alpha = None if alpha is None else alpha[frequency]
    return (None if TL is None else float(TL)), alpha


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def _intersect_parameters(sx, sy, X, Y, wall):
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
    t, u = _intersect_parameters(sx, sy, X, Y, wall)
    return (t > 1e-9) & (t < 1 - 1e-9) & (u >= 0) & (u <= 1)


def _distance(sx, sy, X, Y, r_ref):
    return np.maximum(np.hypot(X - sx, Y - sy), r_ref)


def _knife_edge_loss(delta, wavelength):
    """Positive-path-difference single-edge approximation (not exact 2D diffraction)."""
    v = np.sqrt(np.maximum(0.0, 2 * delta / wavelength))
    return 6.9 + 20 * np.log10(np.sqrt((v - 0.1)**2 + 1) + v - 0.1)


def _reflection_intensity(sx, sy, L_ref, X, Y, wall, r_ref, frequency=1000):
    TL, alpha = material_properties(wall.material, frequency)
    if alpha is None:
        # Missing absorption measurement: reflection is NOT modelled.
        return np.zeros_like(np.asarray(X, dtype=float))

    # Opaque-backed surface approximation: reflection from measured absorption.
    rho = max(0.0, 1.0 - alpha)
    if rho == 0:
        return np.zeros_like(np.asarray(X, dtype=float))

    ax, ay = wall.start
    bx, by = wall.end
    wx, wy = bx - ax, by - ay
    length2 = wx * wx + wy * wy
    if length2 == 0:
        raise ValueError("Wall endpoints must be different")

    q = ((sx - ax) * wx + (sy - ay) * wy) / length2
    px, py = ax + q * wx, ay + q * wy
    mx, my = 2 * px - sx, 2 * py - sy
    t, u = _intersect_parameters(mx, my, X, Y, wall)
    source_side = _cross(wx, wy, sx - ax, sy - ay)
    receiver_side = _cross(wx, wy, X - ax, Y - ay)
    valid = ((t > 0) & (t < 1) & (u >= 0) & (u <= 1)
             & (source_side * receiver_side > 0))
    d = _distance(mx, my, X, Y, r_ref)
    return np.where(valid, rho * 10 ** (L_ref / 10) * (r_ref / d)**2, 0.0)


def source_intensity(sx, sy, L_ref, X, Y, walls=(), frequency=1000, r_ref=1.0):
    """One-wall free-field + transmitted + diffracted + first-order reflected energy."""
    if len(walls) > 1:
        raise NotImplementedError("Use the four-wall script for multiple walls")
    d = _distance(sx, sy, X, Y, r_ref)
    free = 10 ** (L_ref / 10) * (r_ref / d)**2
    if not walls:
        return free

    wall = walls[0]
    TL, _ = material_properties(wall.material, frequency)
    if TL is None:
        raise ValueError("Transmission model requires measured TL for this wall assembly")
    blocked = _blocked(sx, sy, X, Y, wall)
    tau = 10 ** (-TL / 10)
    ax, ay = wall.start
    bx, by = wall.end
    path_a = np.hypot(sx - ax, sy - ay) + np.hypot(X - ax, Y - ay)
    path_b = np.hypot(sx - bx, sy - by) + np.hypot(X - bx, Y - by)
    delta = np.maximum(np.minimum(path_a, path_b) - np.hypot(X - sx, Y - sy), 0.0)
    loss = _knife_edge_loss(delta, 343.0 / frequency)
    screened = free * (tau + 10 ** (-loss / 10))
    direct_or_screened = np.where(blocked, screened, free)
    reflected = _reflection_intensity(sx, sy, L_ref, X, Y, wall, r_ref, frequency)
    return direct_or_screened + reflected