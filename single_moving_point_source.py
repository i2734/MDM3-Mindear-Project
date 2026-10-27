import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 201

x = np.linspace(0, 100, N)
y = np.linspace(0, 100, N)
X, Y = np.meshgrid(x, y)

L_ref = 100
r_ref = 1.0

n_steps = 101
source_x_positions = np.linspace(10, 90, n_steps)
source_y = np.linspace(10, 90, n_steps)

sound_maps = np.zeros((n_steps, N, N))

for t, source_x in enumerate(source_x_positions):
    source_y_t = source_y[t]

    distance = np.sqrt(
        (X - source_x)**2 + (Y - source_y_t)**2)
    distance = np.maximum(distance, r_ref)

    sound_maps[t] = L_ref - 20 * np.log10(distance / r_ref)

centre_frame = np.argmin(np.abs(source_x_positions - 50))
centre_map = sound_maps[centre_frame]

# print("Source position:", source_x_positions[centre_frame], source_y)
# print("Minimum level:", centre_map.min(), "dB")
# print("Maximum level:", centre_map.max(), "dB")
# print("Corner level:", centre_map[0, 0], "dB")
# print("Centre level:", centre_map[N // 2, N // 2], "dB")

fig, ax = plt.subplots(figsize=(8, 6))

levels = np.linspace(63, 100, 25)

contour = ax.contourf(
    X, Y, sound_maps[0],
    levels=levels,
    cmap="viridis",
    extend="min"
)

fig.colorbar(contour, ax=ax, label="Sound pressure level (dB)")

source_marker, = ax.plot(
    [source_x_positions[0]], [source_y[0]],
    "r*", markersize=15
)

title = ax.set_title("Moving Point Source")

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect("equal")


def update(frame):
    global contour

    for collection in contour.get_children():
        collection.remove()

    contour = ax.contourf(
        X, Y, sound_maps[frame],
        levels=levels,
        cmap="viridis",
        extend="min"
    )

    source_marker.set_data(
        [source_x_positions[frame]], [source_y[frame]]
    )

    title.set_text(
        f"Moving Point Source — Frame {frame + 1}/{n_steps}"
    )

    return source_marker, title


animation = FuncAnimation(
    fig,
    update,
    frames=n_steps,
    interval=100,
    blit=False,
    repeat=True
)

# plt.tight_layout()
# plt.show()