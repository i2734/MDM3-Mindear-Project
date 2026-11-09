import numpy as np
import matplotlib.pyplot as plt

# Spatial domain
N = 201
x = np.linspace(0, 100, N)
y = np.linspace(0, 100, N)
X, Y = np.meshgrid(x, y)

# Define sources: (x, y, sound level at 1 m)
sources = [
    (30, 50, 100),
    (70, 50, 120),
    (50, 80, 110)
]

r_ref = 1.0

# Total mean-square pressure relative to the reference pressure
total_intensity = np.zeros_like(X)

for source_x, source_y, L_ref in sources:
    distance = np.sqrt(
        (X - source_x)**2 + (Y - source_y)**2
    )
    distance = np.maximum(distance, r_ref)

    # Convert each source's dB level to relative intensity
    intensity = 10**(
        (L_ref - 20 * np.log10(distance / r_ref)) / 10
    )

    total_intensity += intensity

# Convert combined intensity back to dB
sound_level = 10 * np.log10(total_intensity)

sound_maps = sound_level[np.newaxis, :, :]

source_x_positions = np.array([s[0] for s in sources])
source_y = np.array([s[1] for s in sources])

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, sound_level, levels=30)
plt.colorbar(label="Sound pressure level (dB)")

for source_x, source_y, _ in sources:
    plt.scatter(source_x, source_y, color="red", marker="*", s=120)

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Multiple Stationary Point Sources")
plt.axis("equal")
# plt.tight_layout()
# plt.show()
