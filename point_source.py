import numpy as np
import matplotlib.pyplot as plt

# Define the spatial domain
x_min, x_max = 0, 100
y_min, y_max = 0, 100
N = 201

x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, N)
X, Y = np.meshgrid(x, y)

# Define the stationary sound source position
source_x = 50
source_y = 50

# Sound level at the reference distance 
L_ref = 100  # dB
r_ref = 1    # m (1m radius of sound source produces reference sound level)

# Calculate distance from the source at every grid point
distance = np.sqrt((X - source_x)**2 + (Y - source_y)**2)

# Avoid the singularity at the source
distance = np.maximum(distance, r_ref)

# Calculate the sound pressure level
sound_level = L_ref - 20 * np.log10(distance / r_ref)

sound_maps = sound_level[np.newaxis, :, :]

source_x_positions = np.array([source_x])
source_y = np.array([source_y])

# Plot the reference sound map
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, sound_level, levels=30)
plt.colorbar(label="Sound pressure level (dB)")

plt.scatter(source_x, source_y, color="red", marker="*", s=150)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Single Stationary Point Source")
plt.axis("equal")
# plt.tight_layout()
# plt.show()


