import numpy as np
import matplotlib.pyplot as plt

import distributed_background_noise as sound_map
from extract_sound import prepare_interpolation_data


# 1. Load the sound maps and grid
sound_maps = sound_map.sound_maps
x = sound_map.x
y = sound_map.y

source_x = sound_map.source_x_positions
source_y = sound_map.source_y


# 2. Define phone configurations
phone_counts = [5, 10, 20, 50, 100]
layouts = ["random", "grid", "clustered"]


# 3. Prepare interpolation data (THIS WILL BE INPUT INTO THE INERPOLATION MODELS)
prepared_data = prepare_interpolation_data(
    sound_maps=sound_maps,
    x=x,
    y=y,
    phone_counts=phone_counts,
    layouts=layouts
)

# 4. Inspect output dimensions
print("Sound map shape:", sound_maps.shape)
print("Number of configurations:", len(prepared_data))

# Print one example: 20 phones with a random layout
example = prepared_data[(20, "random")]

print("\nPhone positions:")
print(example["phone_positions"])

print("\nPhone measurements (first 5 time steps):")
print(example["phone_levels"][:5])

print("\nQuery positions (first 10 points):")
print(example["query_positions"][:10])

print("\nTrue sound levels (first time step, first 10 points):")
print(example["true_levels"][0, :10])