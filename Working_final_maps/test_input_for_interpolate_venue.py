""" Example of how to load the sound map and prepare microphone measurements for interpolation."""

# uncomment the appropriate line below to choose the venue sound map
import concert_map as sound_map
# import stadium_map as sound_map

from extract_sound_venues import prepare_interpolation_data


# load the simulated sound map and audience area
sound_maps, x, y, allowed_mask = sound_map.load_sound_map()

# number of phones and layouts to compare
phone_counts = [10, 30, 50, 100, 250]
layouts = ["random", "grid", "clustered"]

# generate phone positions and extract their sound measurements
prepared_data = prepare_interpolation_data(
    sound_maps=sound_maps,
    x=x,
    y=y,
    phone_counts=phone_counts,
    layouts=layouts,
    seed=42,
    allowed_mask=allowed_mask)

# choose which configuration to use for interpolation
n_phones = 50
layout = "grid"

data = prepared_data[(n_phones, layout)]

phone_positions = data["phone_positions"]
phone_levels = data["phone_levels"]
query_positions = data["query_positions"]
true_levels = data["true_levels"]
