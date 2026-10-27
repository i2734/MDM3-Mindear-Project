import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 201
x = np.linspace(0, 100, N)
y = np.linspace(0, 100, N)
X, Y = np.meshgrid(x, y)

sources = [
    (20, 25, 75),
    (45, 25, 80),
    (75, 30, 78),
    (25, 70, 82),
    (60, 70, 76),
    (80, 80, 80)
]

r_ref = 1.0
n_steps = 100
time = np.linspace(0, 2 * np.pi, n_steps)

sound_maps = np.zeros((n_steps, N, N))

for t in range(n_steps):
    total_intensity = np.zeros_like(X)

    for i, (source_x, source_y, L_ref) in enumerate(sources):
        activity_change = 20 * np.sin(time[t] + i)
        current_level = L_ref + activity_change

        distance = np.sqrt(
            (X - source_x)**2 + (Y - source_y)**2
        )
        distance = np.maximum(distance, r_ref)

        intensity = 10**(
            (current_level - 20 * np.log10(distance / r_ref)) / 10
        )

        total_intensity += intensity

    sound_maps[t] = 10 * np.log10(total_intensity)


# Standardised variables for the test code
source_x_positions = np.array([s[0] for s in sources])
source_y = np.array([s[1] for s in sources])


print("First frame range:", sound_maps[0].min(), sound_maps[0].max())
print("Middle frame range:", sound_maps[n_steps // 2].min(), sound_maps[n_steps // 2].max())
print("Last frame range:", sound_maps[-1].min(), sound_maps[-1].max())


fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(
    sound_maps[0],
    extent=[0, 100, 0, 100],
    origin="lower",
    cmap="viridis",
    vmin=40,
    vmax=100,
    interpolation="nearest",
    aspect="equal"
)

fig.colorbar(im, ax=ax, label="Sound pressure level (dB)")

source_x = [s[0] for s in sources]
source_y = [s[1] for s in sources]

markers, = ax.plot(
    source_x, source_y,
    "r.", markersize=8
)

title = ax.set_title("Distributed Background Noise")

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)


def update(frame):
    im.set_data(sound_maps[frame])
    title.set_text(
        f"Distributed Background Noise — Frame {frame + 1}/{n_steps}"
    )
    return im, title


animation = FuncAnimation(
    fig,
    update,
    frames=n_steps,
    interval=100,
    blit=False,
    repeat=True,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()