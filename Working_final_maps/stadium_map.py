"""Crowd noise simulation for an Ashton Gate-inspired stadium.

Uses four stands with diagonal corners, local chanting, goal celebrations and concrete wall reflections.
The stadium dimensions and crowd activity are approximate.
"""
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np

from single_wall import Wall, _intersect_parameters, material_properties

OUT = Path(__file__).resolve().parent
FREQUENCIES = (125, 250, 500, 1000, 2000, 4000)


#verified from Ashton Gate stadium website (transfermarket)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 69.0

RUNOFF = 5.0 #looking at pictures of the pitch 5m seems like a reasonable value

# stadium stand dimensions are calcualted from from Google Earth aerial imagery (metres) of stadium. Modelling assumption: pitch is centred
WEST_DEPTH = 40.5
EAST_DEPTH = 40.5
NORTH_DEPTH = 30.0
SOUTH_DEPTH = 30.0

WIDTH = WEST_DEPTH + RUNOFF + PITCH_WIDTH + RUNOFF + EAST_DEPTH
LENGTH = SOUTH_DEPTH + RUNOFF + PITCH_LENGTH + RUNOFF + NORTH_DEPTH

PITCH_X0 = WEST_DEPTH + RUNOFF
PITCH_Y0 = SOUTH_DEPTH + RUNOFF
PITCH_X1 = PITCH_X0 + PITCH_WIDTH
PITCH_Y1 = PITCH_Y0 + PITCH_LENGTH

# Inner boundary of spectator stands (outside the runoff)
IX0, IX1 = PITCH_X0 - RUNOFF, PITCH_X1 + RUNOFF
IY0, IY1 = PITCH_Y0 - RUNOFF, PITCH_Y1 + RUNOFF

# using a 1m grid to calculate sound levels across the stadium
x = np.arange(0.0, WIDTH + 1.0, 1.0)
y = np.arange(0.0, LENGTH + 1.0, 1.0)
X, Y = np.meshgrid(x, y)

# Four stands with diagonal boundaries (simplified partition) between the stadium corners and the pitch runoff
STANDS = {
    "Lansdown": [(0., 0.), (IX0, IY0), (IX0, IY1), (0., LENGTH)],
    "Dolman": [(WIDTH, 0.), (WIDTH, LENGTH), (IX1, IY1), (IX1, IY0)],
    "Atyeo": [(0., LENGTH), (IX0, IY1), (IX1, IY1), (WIDTH, LENGTH)],
    "South": [(0., 0.), (WIDTH, 0.), (IX1, IY0), (IX0, IY0)],
    }
# convert the stand outlines into paths so we can check which points fall inside them
STAND_PATHS = {name: MplPath(vertices) for name, vertices in STANDS.items()}


def polygon_mask(path, px, py, margin=0.0):
    """Return a boolean mask of points inside a stand, leaving the requested margin."""
    # put x and y coordinates together so we can check all the points
    pts = np.column_stack((np.asarray(px).ravel(), np.asarray(py).ravel()))
    return path.contains_points(pts, radius=-2 * margin).reshape(np.shape(px))


AUDIENCE_MARGIN = 1.0 # adds margin between concrete stadium walls and where audience members can record from/be located
STAND_MASKS = {name: polygon_mask(path, X, Y, AUDIENCE_MARGIN) for name, path in STAND_PATHS.items()}
AUDIENCE_MASK = np.logical_or.reduce(list(STAND_MASKS.values()))

# Four outer concrete walls
WALLS = [
    ("South wall", (0., 0.), (WIDTH, 0.), "concrete"),
    ("East wall", (WIDTH, 0.), (WIDTH, LENGTH), "concrete"),
    ("North wall", (WIDTH, LENGTH), (0., LENGTH), "concrete"),
    ("West wall", (0., LENGTH), (0., 0.), "concrete"),]

# Sound power per spectator at each frequency, using Shubin et al. (2021) data
SPECTATOR_LW_DB = np.array([68.0, 63.8, 73.6, 77.6, 70.3, 63.1])

#verified from ashton gate stadium website (transfermarket)
ASSUMED_OCCUPANCY = 27000 

SOURCE_SPACING = 8.0 #(across stands) design choice: each source represents 8mx8m space, when divided by the total stand area it gives as just under 300 sound sources with each source representing around 91 spectators
SOURCE_SOFTENING_M = 2.0  # Avoids very large values close to a source

#simulating 3 minutes of football match activity 
DURATION_SECONDS = 3 * 60 

#each frame respresents 2 seconds of match time
TIME_STEP_SECONDS = 2.0
N_STEPS = int(round(DURATION_SECONDS / TIME_STEP_SECONDS)) + 1
SEED = 42
SECTIONS_PER_STAND = 6 #6 sections per stand, so 24 sections across the 4 stands
# when chanting occurs in a single section, the sound sources (that are randomly scattered across the stands), in that section will light up 

CHANT_BOOST_DB = 6.0  # Assumed lower-intensity crowd activity than goal boost
CHANT_DURATION_SECONDS = 30.0  # Assumed duration of a chant
CHANTS_PER_STAND = 2
GOAL_BOOST_DB = 7.0  # Cheshunt Sports Village assessment: 56 to 63 dB during cheering
GOAL_DURATION_SECONDS = 30.0  # Wembley Stadium planning report (2019)

# one goal 2 minutes into the 3 minutes of simulated match time 
GOAL_TIMES = (120,)
numerical_tolerance = 1e-8


# convert dB to linear energy because we can't add sound levels in dB
def energy_from_db(db):
    """Convert dB to linear energy so contributions can be added."""
    return 10.0 ** (np.asarray(db) / 10.0)


# convert the total energy back into dB
def db_from_energy(energy):
    """Convert summed linear energy back to dB, avoiding log of zero."""
    return 10.0 * np.log10(np.maximum(energy, 1e-30))


def wall_object(wall):
    """Turn a wall's two endpoints into the geometry object used for intersections."""
    return Wall(wall[1], wall[2])


@lru_cache(maxsize=None)
def concrete_reflectivity(frequency):
    """Get the fraction of energy reflected by concrete at this frequency (1 - absorption)."""
    _, alpha = material_properties("concrete", frequency)
    if alpha is None:
        raise ValueError(f"No concrete absorption coefficient at {frequency} Hz")
    # assuming all the energy that is not absorbed gets reflected
    return float(np.clip(1.0 - alpha, 0.0, 1.0))


# calculate each stand area to work out how many sound sources to use
def polygon_area(vertices):
    """Calculate stand area from its corner coordinates using the shoelace formula."""
    p = np.asarray(vertices, dtype=float)
    return abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2.0


def crowd_sources():
    """Place roughly one source per 8 m square in each stand and assign its chanting section."""
    rng = np.random.default_rng(SEED)
    sources = []
    # make a list of the 24 sections across the 4 stands
    sections = [(stand, k) for stand in STANDS for k in range(SECTIONS_PER_STAND)]
    section_lookup = {key: i for i, key in enumerate(sections)}

    for stand, vertices in STANDS.items():
        verts = np.asarray(vertices)
        xmin, ymin = verts.min(axis=0)
        xmax, ymax = verts.max(axis=0)
        # divide the stand area by 8x8 to decide how many sources to generate
        count = max(1, round(polygon_area(vertices) / SOURCE_SPACING**2))
        path = STAND_PATHS[stand]
        accepted = 0
        attempts = 0
        while accepted < count:
            attempts += 1
            if attempts > count * 1000:
                raise RuntimeError(f"Could not sample enough sources in {stand}")
            # generate random coordinates, then reject points outside the stand
            sx = rng.uniform(xmin, xmax)
            sy = rng.uniform(ymin, ymax)
            if not polygon_mask(path, np.array([sx]), np.array([sy]), AUDIENCE_MARGIN)[0]:
                continue
            # Divide side stands along their length and end stands across their width
            along = (sy - ymin) / (ymax - ymin) if stand in ("Lansdown", "Dolman") else (sx - xmin) / (xmax - xmin)
            # the section depends on the source position, it is not assigned randomly
            section = min(int(along * SECTIONS_PER_STAND), SECTIONS_PER_STAND - 1)
            sources.append((sx, sy, section_lookup[(stand, section)]))
            accepted += 1
    return sources, sections


SOURCES, SECTIONS = crowd_sources()


# inverse square spreading, with the 2m term stopping very high levels near a source
def direct_energy(sx, sy):
    """Calculate inverse-square spreading from one source across the grid, softened within 2 m."""
    return 1.0 / ((X - sx)**2 + (Y - sy)**2 + SOURCE_SOFTENING_M**2)


# mirror the source across the wall to calculate where reflected sound comes from
def mirror_point(px, py, wall):
    """Find the image of a source across a wall for the reflection calculation."""
    ax, ay = wall[1]
    bx, by = wall[2]
    wx, wy = bx - ax, by - ay
    # find the position along the wall line closest to the source
    q = ((px - ax) * wx + (py - ay) * wy) / (wx * wx + wy * wy)
    return 2 * (ax + q * wx) - px, 2 * (ay + q * wy) - py


def intersection(px, py, qx, qy, wall):
    """Find where a path hits a finite wall and flag intersections within both segments."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t, u = _intersect_parameters(px, py, qx, qy, wall_object(wall))
        hx = px + t * (qx - px)
        hy = py + t * (qy - py)
    # only keep intersections that actually hit the wall segment
    valid = (np.isfinite(t) & np.isfinite(u) & (t > numerical_tolerance) & (t < 1.0 - numerical_tolerance) & (u >= -numerical_tolerance) & (u <= 1.0 + numerical_tolerance))
    return hx, hy, valid


# check whether two points are on the same side of a wall
def same_side(px, py, qx, qy, wall):
    """Check that the source and receiver lie on the same side of a reflecting wall."""
    ax, ay = wall[1]
    bx, by = wall[2]
    wx, wy = bx - ax, by - ay
    side_a = wx * (py - ay) - wy * (px - ax)
    side_b = wx * (qy - ay) - wy * (qx - ax)
    return side_a * side_b > numerical_tolerance


def reflected_geometry(sx, sy, wall_sequence):
    """Trace one- or two-bounce paths using image sources and return their distance losses."""
    # create mirror sources for the walls the sound reflects off
    images = [(sx, sy)]
    for wall in wall_sequence:
        images.append(mirror_point(*images[-1], wall))

    # work backwards from the receiver to find the reflection points
    bounces = [None] * len(wall_sequence)
    next_x, next_y = X, Y
    valid = np.ones_like(X, dtype=bool)

    # Check where the reflected path hits each finite wall
    for j in range(len(wall_sequence) - 1, -1, -1):
        hx, hy, ok = intersection(*images[j + 1], next_x, next_y, wall_sequence[j])
        bounces[j] = (hx, hy)
        valid &= ok
        next_x, next_y = hx, hy

    path = [(sx, sy)] + bounces + [(X, Y)]
    # add up the lengths of each part of the reflected sound path
    distance = np.zeros_like(X, dtype=float)
    for a, b in zip(path[:-1], path[1:]):
        distance += np.hypot(b[0] - a[0], b[1] - a[1])

    for j, wall in enumerate(wall_sequence):
        valid &= same_side(*path[j], *path[j + 2], wall)

    # Diagonal stand boundaries are not physical walls, so they do not reflect sound
    return np.where(valid & np.isfinite(distance), 1.0 / (distance**2 + SOURCE_SOFTENING_M**2), 0.0)


@lru_cache(maxsize=2)
def section_propagation(reflections=True):
    """Add direct and reflected sound from all sources into section-by-frequency maps."""
    # store the sound field for each section and frequency
    fields = np.zeros((len(SECTIONS), len(FREQUENCIES), len(y), len(x)), dtype=np.float32)
    # Calculate reflection paths once per source, then apply absorption at each frequency
    for sx, sy, section in SOURCES:
        # Sound energy decreases with distance squared
        direct = direct_energy(sx, sy)
        first = [reflected_geometry(sx, sy, (wall,)) for wall in WALLS] if reflections else []
        second = []
        if reflections:
            # test pairs of different walls for second-order reflections
            for w1 in WALLS:
                for w2 in WALLS:
                    if w1 is not w2:
                        second.append((w1, w2, reflected_geometry(sx, sy, (w1, w2))))

        for fi, frequency in enumerate(FREQUENCIES):
            field = direct.copy()
            if reflections:
                # each reflection reduces the energy by the concrete reflection factor
                rho = concrete_reflectivity(frequency)
                for geom in first:
                    field += rho * geom
                for _, _, geom in second:
                    field += rho * rho * geom
            fields[section, fi] += field.astype(np.float32)
    return fields


# assumed event shape: 2s rise, stays loud, then fades over the final 30%
def event_envelope(t, onset, duration):
    """Give a chant or goal a 2 s rise, steady middle and fade over its final 30%."""
    elapsed = t - onset
    attack = np.clip(elapsed / 2.0, 0.0, 1.0)
    decay = np.clip((duration - elapsed) / max(duration * 0.3, 0.01), 0.0, 1.0)
    return np.where((elapsed >= 0) & (elapsed <= duration), np.minimum(attack, decay), 0.0)


def crowd_activity(n_steps=N_STEPS, duration_seconds=DURATION_SECONDS):
    """Build the extra dB for each crowd section over time from chants and goals."""
    rng = np.random.default_rng(SEED + 1)
    t = np.linspace(0.0, duration_seconds, n_steps)
    # extra dB for every section at every time step, initially zero
    activity = np.zeros((n_steps, len(SECTIONS)), dtype=float)
    lookup = {key: i for i, key in enumerate(SECTIONS)}
    # Each stand has independent chanting events
    if CHANT_BOOST_DB:
        for stand in STANDS:
            for _ in range(CHANTS_PER_STAND):
                # randomly choose the chanting section and when its chant starts
                section = int(rng.integers(0, SECTIONS_PER_STAND))
                onset = float(rng.uniform(0.0, max(0.0, duration_seconds - CHANT_DURATION_SECONDS)))
                activity[:, lookup[(stand, section)]] += CHANT_BOOST_DB * event_envelope(t, onset, CHANT_DURATION_SECONDS)

    # Goal celebrations affect the entire stadium
    if GOAL_BOOST_DB:
        for onset in GOAL_TIMES:
            activity += (GOAL_BOOST_DB * event_envelope(t, onset, GOAL_DURATION_SECONDS))[:, None]
    return t, activity


@lru_cache(maxsize=4)
def simulate(reflections=True, n_steps=N_STEPS):
    """Combine crowd sound powers, propagation and activity into time-varying SPL maps."""
    fields = section_propagation(reflections)
    # each equivalent source represents an equal number of spectators
    people_per_source = ASSUMED_OCCUPANCY / len(SOURCES)
    times, activity_db = crowd_activity(n_steps=n_steps)
    # Combine spectator sound powers, then account for spherical spreading
    source_band_power = energy_from_db(SPECTATOR_LW_DB + 10.0 * np.log10(people_per_source) - 10.0 * np.log10(4.0 * np.pi))
    broadband = np.empty((n_steps, len(y), len(x)), dtype=np.float32)
    bands = {f: np.empty_like(broadband) for f in FREQUENCIES}
    # Sum the sound energy from all crowd sections at each time step
    for ti in range(n_steps):
        # convert the chant and goal boosts into energy multipliers
        section_power = energy_from_db(activity_db[ti])
        # add the contributions from all sections at each frequency and grid point
        energies = np.einsum('s,sfyx,f->fyx', section_power, fields, source_band_power, optimize=True)
        # combine the six frequency bands into an unweighted broadband level
        broadband[ti] = db_from_energy(energies.sum(axis=0))
        for fi, f in enumerate(FREQUENCIES):
            bands[f][ti] = db_from_energy(energies[fi])
    return broadband, bands


def load_sound_map(reflections=True, n_steps=N_STEPS):
    """Return the simulated maps, grid coordinates and valid spectator positions."""
    sound_maps, _ = simulate(reflections=reflections, n_steps=n_steps)
    return sound_maps.copy(), x.copy(), y.copy(), AUDIENCE_MASK.copy()


def draw_venue(ax, show_sources=False):
    """Draw stand outlines, pitch and walls, optionally showing the crowd source locations."""
    for name, vertices in STANDS.items():
        ax.add_patch(Polygon(vertices, closed=True, fill=False, edgecolor="white", linewidth=1.4, zorder=5))
        verts = np.asarray(vertices)
        cx, cy = verts.mean(axis=0)
        ax.text(cx, cy, name.upper(), ha="center", va="center", color="white", fontsize=8, weight="bold", zorder=7, bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2))

    # Draw the runoff area and pitch over the sound map
    ax.add_patch(Rectangle((IX0, IY0), IX1 - IX0, IY1 - IY0, facecolor="#dddddd", edgecolor="white", linewidth=1, zorder=3))
    ax.add_patch(Rectangle((PITCH_X0, PITCH_Y0), PITCH_WIDTH, PITCH_LENGTH, facecolor="#3d8b55", edgecolor="white", linewidth=1.5, zorder=4))
    ax.plot([PITCH_X0, PITCH_X1], [(PITCH_Y0 + PITCH_Y1) / 2] * 2, color="white", lw=1, zorder=5)
    ax.add_patch(Circle(((PITCH_X0 + PITCH_X1) / 2, (PITCH_Y0 + PITCH_Y1) / 2), 9.15, fill=False, edgecolor="white", lw=1, zorder=5))
    for _, a, b, _ in WALLS:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=2.5, zorder=6)
    if show_sources:
        pts = np.array([(sx, sy) for sx, sy, _ in SOURCES])
        ax.scatter(pts[:, 0], pts[:, 1], s=4, c="white", alpha=0.4, zorder=6)
    ax.set(xlim=(-3, WIDTH + 3), ylim=(-3, LENGTH + 3), xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal")


def main():
    """Animate SPL in the stands using one colour scale across all frames."""
    sound_maps, _ = simulate(reflections=True)
    # only display sound levels in the spectator areas
    display_maps = np.ma.masked_where(~np.broadcast_to(AUDIENCE_MASK, sound_maps.shape), sound_maps)
    # use the same colour scale for all frames so changes can be compared
    vmin, vmax = float(display_maps.min()), float(display_maps.max())
    fig, ax = plt.subplots(figsize=(8, 10), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(display_maps[0], extent=(x[0], x[-1], y[0], y[-1]), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    draw_venue(ax)
    fig.colorbar(im, ax=ax, label="Sound Pressure Level (dB)", shrink=0.75, pad=0.02)
    title = ax.set_title("Stadium Sound Map (Ashton Gate-inspired stadium) — frame 1")

    def update(frame):
        """Replace the heatmap and title with the next frame of the simulation."""
        im.set_data(display_maps[frame])
        title.set_text(f"Stadium Sound Map (Ashton Gate-inspired stadium) — frame {frame + 1}/{len(sound_maps)}")
        return im, title

    # 120ms is the animation playback speed, not the 2s of match time per frame
    animation = FuncAnimation(fig, update, frames=len(sound_maps), interval=120, blit=False, repeat=True, cache_frame_data=False)
    fig._stadium_animation = animation

    levels = sound_maps[:, AUDIENCE_MASK]
    # comparing the overall max and min with Veritas Hearing's "FIFA World Cup 2026: How Loud Will Stadiums Get?" shows results are aligned.
    print(f"Overall minimum: {levels.min():.1f} dB")
    print(f"Overall maximum: {levels.max():.1f} dB")


    plt.show()

    

if __name__ == "__main__":
    main()
