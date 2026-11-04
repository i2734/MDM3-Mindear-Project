
"""
SoundTracker — interpolation input and microphone visualisation.

The venue model defines the audience area.
This script controls microphone numbers and layouts.
"""

import numpy as np
import matplotlib.pyplot as plt

import concert_map as sound_map

from extract_sound_venues import prepare_interpolation_data


# ============================================================
# 1. LOAD SOUND MAP
# ============================================================

sound_maps, x, y, allowed_mask = sound_map.load_sound_map()


# ============================================================
# 2. CHOOSE MICROPHONE CONFIGURATIONS
# ============================================================

phone_counts = [10, 30, 50, 100, 250]

layouts = ["random", "grid", "clustered"]


# ============================================================
# 3. PREPARE MEASUREMENTS
# ============================================================

prepared_data = prepare_interpolation_data(
    sound_maps=sound_maps,
    x=x,
    y=y,
    phone_counts=phone_counts,
    layouts=layouts,
    seed=42,
    allowed_mask=allowed_mask
)


# ============================================================
# 4. SELECT CONFIGURATION FOR INTERPOLATION
# ============================================================

# n_phones = 20
# layout = "grid"

# data = prepared_data[(n_phones, layout)]

# phone_positions = data["phone_positions"]
# phone_levels = data["phone_levels"]

# query_positions = data["query_positions"]
# true_levels = data["true_levels"]

# print("Sound map shape:", np.asarray(sound_maps).shape)
# print("Configurations:", len(prepared_data))
# print("Phones:", phone_positions.shape)
# print("Measurements:", phone_levels.shape)
# print("Queries:", query_positions.shape)
# print("Truth:", true_levels.shape)
